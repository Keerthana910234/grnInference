# %% [markdown]
# # Code to simulate a synthetic GRN and infer the network using TwINFER
# 

# %% [markdown]
# ## Details about the simulation
# 
# ### Set this for both running the simulation and before inferring using TwINFER
# 

# %%
import sys
from pathlib import Path
path_to_code_repo = Path("/home/gzu5140/Keerthana_b1042/grnInference/code/grnInferenceRepo/")
sys.path.append(str(path_to_code_repo))


# %%
path_to_data = "/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/median_parameter_simulations"

base_config = {
    'n_cells': 6000, #Number of cells before division (number of twin pairs)
    'simulation_time_before_division': 2000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
    'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
    'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
    "path_to_connectivity_matrix": f"{path_to_data}/figure_2_simulation_details/interaction_matrix_A_rep_B_B_to_A.txt", #path to the connectivity matrix specifying the GRN to simulate
    "param_csv": f"{path_to_data}/figure_2_simulation_details/median_param.csv", #Path to the parameters for all genes and interaction terms
    "rows_to_use": [[7,7]], #Rows in the parameter's csv file for each gene - the length should be equal to number of genes in the system
    "output_folder": f"{path_to_data}/figure_3_simulations/A_rep_B_B_to_A/", #Path to folder to store simulation 
    "log_file": f"{path_to_data}/figure_2_simulation_details/fig_2_median_parameter_simulations_6000_cells.jsonl", #Path to the log file
    "type": "A_rep_B_B_to_A",  # Name of the network used -- will be in the filename
    "number_parallel_processes": 1, #Number of parameters to be run in parallel
    "number_of_cores_per_parameter": 20, #Number of cores to be used per parameter (number_parallel_processes * number_of_cores_per_parameter = number of cores in your computer)
}

# %% [markdown]
# ## Functions and packages used in this notebook - run this everytime notebook is restarted
# 

# %% [markdown]
# ### For the simulation
# 

# %%
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import os
import sys
import numpy as np
from numba import set_num_threads, get_num_threads
set_num_threads(base_config['number_of_cores_per_parameter'])
print("Threads Numba will use:", get_num_threads())

import importlib
from simulation_scripts.gillespie_simulation import gillespie_script

importlib.reload(gillespie_script)


from simulation_scripts.gillespie_simulation.gillespie_script import process_param_set



# %% [markdown]
# ### For inferring with TwINFER
# 

# %%
# Calculation functions
import importlib
from simulation_scripts.gillespie_simulation import correlation_analysis_functions
from simulation_scripts.gillespie_simulation import correlation_analysis_helpers

importlib.reload(correlation_analysis_functions)
importlib.reload(correlation_analysis_helpers)

from simulation_scripts.gillespie_simulation.correlation_analysis_functions import (
    calculate_pairwise_gene_gene_correlation_matrix,
    check_system_in_steady_state,
    check_gene_gene_correlation_threshold,
    calculate_twin_random_pair_correlations,
    differentiate_single_state_reg_and_multiple_states,
    identify_reg_if_multiple_states,
    get_directions_from_simulation
)

# Helper functions
from simulation_scripts.gillespie_simulation.correlation_analysis_helpers import (
    extract_param_index,
    read_input_matrix,
    get_param_data, 
    plot_matrix_as_heatmap,
    print_summary,
    plot_network
)

import pandas as pd
import matplotlib.pyplot as plt

# %%
def infer_using_twinfer(path_to_simulation_file, base_config, t1, t2, 
                        check_for_steady_state=True, 
                        plot_correlation_matrices_as_heatmap=True, have_any_output = True):
    """
    Perform twin-based inference of gene regulatory interactions from simulation data.

    This function analyzes simulation results to identify gene-gene regulatory relationships,
    distinguish between single- and multi-state regulation, and infer directionality using
    time-shifted correlations between twin cells.

    Parameters
    ----------
    path_to_simulation_file : str
        Path to the CSV file containing simulation output (one clone per row per timepoint).
    
    base_config : dict
        Dictionary containing simulation metadata and parameters:
            - "n_cells": expected number of twin clones
            - "twin_simulation_time_after_division": total duration after division (hours)
            - "twin_measurement_resolution": sampling resolution (hours)
            - "path_to_connectivity_matrix": file path to interaction matrix
            - "param_csv": file path to parameter CSV
            - "rows_to_use": list of row index sets used in this simulation

    t1 : int
        Timepoint (in hours) for early gene expression correlation.

    t2 : int
        Timepoint (in hours) for late-stage twin/random cell analysis.

    check_for_steady_state : bool, default=True
        If True, checks that simulations have reached steady state by t1.

    plot_correlation_matrices_as_heatmap : bool, default=True
        If True, visualizes intermediate correlation matrices using heatmaps.

    Returns
    -------
    dict
        Dictionary of inference results:
            - "direction_matrix": directional correlation matrix (t1 → t2)
            - "direction_raw_matrix": raw correlation difference matrix (no threshold)
            - "pairwise_gene_gene_correlation_matrix": gene-gene correlations at t1
            - "twin_pair_correlation_matrix_t2": twin correlation matrix at t2
            - "random_pair_correlation_matrix_t2": random correlation matrix at t2
            - "twin_pair_correlation_matrix_t1": twin correlation matrix at t1 (if used)
            - "random_pair_correlation_matrix_t1": random correlation matrix at t1 (if used)

    Notes
    -----
    - Regulation type is classified as either single-state or multi-state using twin vs random correlations.
    - Directionality is inferred only for single-state interactions.
    - Intermediate checks ensure simulation structure and parameter identity match the base_config.
    """


    # Load simulation data
    try:
        simulation = pd.read_csv(path_to_simulation_file)
    except Exception as e:
        raise RuntimeError(f"Error reading the simulation file: {e}")

    # Load connectivity matrix and parameter set
    path_to_connectivity_matrix = base_config["path_to_connectivity_matrix"]
    path_to_parameter_csv = base_config["param_csv"]
    param_df = pd.read_csv(path_to_parameter_csv, index_col=0)

    # --- Basic sanity checks ---
    # Assert number of clones in simulation file matches config
    n_clones_simulation = simulation['clone_id'].nunique()
    n_clones_base_config = base_config["n_cells"]
    assert n_clones_simulation == n_clones_base_config, \
        "Number of twin pairs in the simulation file does not match n_cells in base_config."

    # Assert time points match expected resolution
    time_points_simulations = simulation['time_step'].unique()
    time_points_base_config = np.arange(
        0, 
        base_config['twin_simulation_time_after_division'] + base_config['twin_measurement_resolution'], 
        base_config['twin_measurement_resolution']
    )

    assert set(time_points_simulations) == set(time_points_base_config), \
        "The sampling time points in the simulation file do not match those specified in base_config."

    # Assert parameter row identity matches
    param_index_from_file_name = extract_param_index(path_to_simulation_file)
    param_index_from_base_config = "_".join(map(str, base_config["rows_to_use"][0]))
    assert param_index_from_file_name == param_index_from_base_config, \
        "Simulation must match the details in base_config."

    # Load gene parameters and connectivity structure
    gene_params = get_param_data(param_df, param_index_from_file_name)
    n_genes, interaction_matrix = read_input_matrix(path_to_connectivity_matrix)
    gene_list = [f"gene_{i}" for i in np.arange(1, n_genes + 1)]

    
    # --- Check for steady state at t1 (optional) ---
    if check_for_steady_state:
        is_system_in_steady_state = check_system_in_steady_state(simulation, gene_params, interaction_matrix, gene_list,
                                  relative_diff_threshold=0.05, relative_slope_threshold=0.01)
        if not is_system_in_steady_state:
            raise ValueError(
                "The system is not in steady state. "
                "You can override this by setting check_for_steady_state=False."
            )

    # Ensure the time points t1 and t2 exist in the simulation data
    unique_timepoints = simulation['time_step'].unique()

    if t1 not in unique_timepoints:
        raise ValueError(f"Time point t1={t1} not found in simulation['time_step'].")
    if t2 not in unique_timepoints:
        raise ValueError(f"Time point t2={t2} not found in simulation['time_step'].")

    # Subset the simulation at the desired timepoints

    # Shuffle all clone IDs
    clone_ids_shuffled = np.random.permutation(n_clones_simulation)

    # Split into 1:1:2 ratio
    n1 = n2 = n_clones_simulation // 4
    t1_clones = clone_ids_shuffled[:n1]
    t2_clones = clone_ids_shuffled[n1:n1 + n2]
    across_t_clones = clone_ids_shuffled[n1 + n2:]

    # Subset directly
    t1_twins = simulation[(simulation['clone_id'].isin(t1_clones)) & (simulation['time_step'] == t1)]
    t2_twins = simulation[simulation['clone_id'].isin(t2_clones) & (simulation['time_step'] == t2)]

    # Across_t: pick exactly one random twin per clone_id
    # One cell per clone at t1
    across_t_twin1 = (
        simulation[(simulation['clone_id'].isin(across_t_clones)) & (simulation['time_step'] == t1)]
        .groupby('clone_id', group_keys=False)
        .sample(n=1, random_state=None)   # set an int for reproducibility
    )

    # One (different) cell from the SAME clones at t2
    candidates_t2 = simulation[(simulation['clone_id'].isin(across_t_clones)) & (simulation['time_step'] == t2)]

    # Exclude the exact cell_ids picked at t1 (if cell_id persists across time)
    candidates_t2 = candidates_t2[~candidates_t2['cell_id'].isin(across_t_twin1['cell_id'])]

    # Now sample one per clone
    across_t_twin2 = (
        candidates_t2
        .groupby('clone_id', group_keys=False)
        .sample(n=1, random_state=None)
    )


    # Reset index for cleanliness
    t1_twins = t1_twins.reset_index(drop=True)
    t2_twins = t2_twins.reset_index(drop=True)
    across_t_twin1 = across_t_twin1.reset_index(drop=True)
    across_t_twin2 = across_t_twin2.reset_index(drop=True)

    all_t1_t2_measurements = pd.concat(
    [t1_twins, t2_twins, across_t_twin1, across_t_twin2],
    ignore_index=True
    )

    
    # --- Step 1: Pairwise gene-gene correlations at t1 ---
    pairwise_gene_gene_correlation_matrix = calculate_pairwise_gene_gene_correlation_matrix(
        all_t1_t2_measurements, gene_list
    )
    scrambled_baseline = calculate_scrambled_baseline_gene_gene_correlation_matrix(
        all_t1_t2_measurements, gene_list
    )
    no_regulation, potential_regulation = check_gene_gene_correlation_threshold(
        pairwise_gene_gene_correlation_matrix, gene_list
    )
    # print(no_regulation)
    if plot_correlation_matrices_as_heatmap:
        plot_matrix_as_heatmap(corr_matrix=pairwise_gene_gene_correlation_matrix, gene_list=gene_list, no_regulation=no_regulation, potential_regulation=potential_regulation,
            title=f"Gene-gene correlations at time {t1}h", add_gene_labels=True, add_time=False, gray_out_no_reg=False
        )

    # --- Step 2: Twin/random correlations at t2 ---
    twin_pair_correlation_matrix_t2, random_pair_correlation_matrix_t2 = calculate_twin_random_pair_correlations(
        all_t1_t2_measurements, t2_twins, gene_list
    )
    # print(twin_pair_correlation_matrix_t2)
    if plot_correlation_matrices_as_heatmap:
        plot_matrix_as_heatmap( corr_matrix=twin_pair_correlation_matrix_t2, gene_list=gene_list, no_regulation=no_regulation, potential_regulation=potential_regulation,
            title=f"Twin pair correlations at time {t2}h", add_gene_labels=True, add_time=True, time=[t2], gray_out_no_reg=True
        )
        
        plot_matrix_as_heatmap( corr_matrix=random_pair_correlation_matrix_t2, gene_list=gene_list, no_regulation=no_regulation, potential_regulation=potential_regulation,
            title=f"Random pair correlations at time {t2}h", add_gene_labels=True, add_time=True, time=[t2], gray_out_no_reg=True
        )

    # --- Step 3: Classify regulation type: single-state vs multiple-states ---
    multiple_states_gene_pairs, single_state_regulation = differentiate_single_state_reg_and_multiple_states(
        potential_regulation, twin_pair_correlation_matrix_t2, random_pair_correlation_matrix_t2, gene_list
    )
    twin_pair_correlation_matrix_t1, random_pair_correlation_matrix_t1 = calculate_twin_random_pair_correlations(
                all_t1_t2_measurements, t1_twins, gene_list
            )
    if len(multiple_states_gene_pairs) > 0:
        
        multiple_states_no_reg, multiple_states_and_reg = identify_reg_if_multiple_states(
            twin_pair_correlation_matrix_t1,twin_pair_correlation_matrix_t2,random_pair_correlation_matrix_t1,
            random_pair_correlation_matrix_t2,multiple_states_gene_pairs,gene_list
            )
    else:
        multiple_states_no_reg, multiple_states_and_reg = [], []

    # --- Step 4: Print summary of results ---
    if have_any_output:
        print_summary(no_regulation, single_state_regulation, multiple_states_no_reg, multiple_states_and_reg)
    
    # --- Step 5: Infer directionality of single-state interactions ---

    direction_raw_matrix, direction_normalized_matrix, pre_threshold_direction_matrix = get_directions_from_simulation(across_t_twin1, across_t_twin2, gene_pairs=single_state_regulation, threshold=None)
    # print(pre_threshold_direction_matrix)

    if plot_correlation_matrices_as_heatmap:
        # Mark all (i, j) entries where direction could not be determined
        undirected_pairs = [
            (g1, g2)
            for g1 in direction_normalized_matrix.index
            for g2 in direction_normalized_matrix.columns
            if (g1, g2) not in single_state_regulation
        ]

        plot_matrix_as_heatmap(
            corr_matrix=direction_normalized_matrix,
            gene_list=list(direction_normalized_matrix.index),
            no_regulation=undirected_pairs,                   
            potential_regulation=single_state_regulation,     
            title=f"Directional correlations (from {t1}h to {t2}h)",
            add_gene_labels=True,
            add_time=True,
            time=[t1, t2],
            gray_out_no_reg=True
        )

    # --- Step 6: Visualize the inferred network ---
    if (len(single_state_regulation) > 0):
        if have_any_output:
            plot_network(direction_normalized_matrix, gene_list)

    return {
        "direction_matrix": direction_normalized_matrix, 
        "direction_raw_matrix": direction_raw_matrix, 
        "pairwise_gene_gene_correlation_matrix": pairwise_gene_gene_correlation_matrix,
        "twin_pair_correlation_matrix_t2": twin_pair_correlation_matrix_t2,
        "random_pair_correlation_matrix_t2": random_pair_correlation_matrix_t2,
        "twin_pair_correlation_matrix_t1": twin_pair_correlation_matrix_t1,
        "random_pair_correlation_matrix_t1": random_pair_correlation_matrix_t1
    }

# %% [markdown]
# ## Simulate the gene expression in a population of cells
# 
# The code simulates gene expression based on a GRN (described by the interaction matrix) and expression of each gene is defined by parameters (each row in the parameter sheet) using the Gillespie algorithm.
# 

# %%
import copy
from tqdm.auto import tqdm

for i in tqdm(np.arange(7,20)):
    run_config = copy.deepcopy(base_config)
    run_config["type"] = f"{run_config['type']}_rep_{i}"

    os.makedirs(run_config['output_folder'], exist_ok=True)
    
    rows_to_use = run_config['rows_to_use']
    labels = ["rows_" + "_".join(map(str, row)) for row in rows_to_use]
    param_sets = list(zip(rows_to_use, labels))

    # Run the simulation for this replicate
    results = Parallel(n_jobs=run_config['number_parallel_processes'])(
        delayed(process_param_set)(rows, label, run_config)
        for rows, label in param_sets
    )

    for (rows, label), res in zip(param_sets, results):
        print(f"[Rep {i}] Completed simulation for {label} (rows={rows}): {res}")

#%%

# import copy
# from tqdm.auto import tqdm

# path_to_data = "/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/median_parameter_simulations/"

# base_config = {
#     'n_cells': 6000, #Number of cells before division (number of twin pairs)
#     'simulation_time_before_division': 1000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
#     'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
#     'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
#     "path_to_connectivity_matrix": f"{path_to_data}/figure_2_simulation_details/interaction_matrix_A_and_B.txt", #path to the connectivity matrix specifying the GRN to simulate
#     "param_csv": f"{path_to_data}/figure_2_simulation_details/median_param.csv", #Path to the parameters for all genes and interaction terms
#     "rows_to_use": [[0,0]], #Rows in the parameter's csv file for each gene - the length should be equal to number of genes in the system
#     "output_folder": f"{path_to_data}/figure_3_simulations/A_to_B_B_to_A/", #Path to folder to store simulation 
#     "log_file": f"{path_to_data}/figure_2_simulation_details/fig_2_median_parameter_simulations_6000_cells.jsonl", #Path to the log file
#     "type": "A_and_B",  # Name of the network used -- will be in the filename
#     "number_parallel_processes": 1, #Number of parameters to be run in parallel
#     "number_of_cores_per_parameter": 24, #Number of cores to be used per parameter (number_parallel_processes * number_of_cores_per_parameter = number of cores in your computer)
# }

# for i in tqdm(range(19)):
#     run_config = copy.deepcopy(base_config)
#     run_config["type"] = f"{run_config['type']}_rep_{i}"

#     os.makedirs(run_config['output_folder'], exist_ok=True)
    
#     rows_to_use = run_config['rows_to_use']
#     labels = ["rows_" + "_".join(map(str, row)) for row in rows_to_use]
#     param_sets = list(zip(rows_to_use, labels))

#     # Run the simulation for this replicate
#     results = Parallel(n_jobs=run_config['number_parallel_processes'])(
#         delayed(process_param_set)(rows, label, run_config)
#         for rows, label in param_sets
#     )

#     for (rows, label), res in zip(param_sets, results):
#         print(f"[Rep {i}] Completed simulation for {label} (rows={rows}): {res}")

#  #%%
# import copy
# from tqdm.auto import tqdm

# path_to_data = "/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/median_parameter_simulations/"

# base_config = {
#     'n_cells': 6000, #Number of cells before division (number of twin pairs)
#     'simulation_time_before_division': 1000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
#     'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
#     'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
#     "path_to_connectivity_matrix": f"{path_to_data}/figure_2_simulation_details/interaction_matrix_A_rep_B_B_to_A.txt", #path to the connectivity matrix specifying the GRN to simulate
#     "param_csv": f"{path_to_data}/figure_2_simulation_details/median_param.csv", #Path to the parameters for all genes and interaction terms
#     "rows_to_use": [[7,7]], #Rows in the parameter's csv file for each gene - the length should be equal to number of genes in the system
#     "output_folder": f"{path_to_data}/figure_3_simulations/A_rep_B_B_to_A/", #Path to folder to store simulation 
#     "log_file": f"{path_to_data}/figure_2_simulation_details/fig_2_median_parameter_simulations_6000_cells.jsonl", #Path to the log file
#     "type": "A_rep_B_B_to_A",  # Name of the network used -- will be in the filename
#     "number_parallel_processes": 1, #Number of parameters to be run in parallel
#     "number_of_cores_per_parameter": 24, #Number of cores to be used per parameter (number_parallel_processes * number_of_cores_per_parameter = number of cores in your computer)
# }

# for i in tqdm(np.arange(0,6)):
#     run_config = copy.deepcopy(base_config)
#     run_config["type"] = f"{run_config['type']}_rep_{i}"

#     os.makedirs(run_config['output_folder'], exist_ok=True)
    
#     rows_to_use = run_config['rows_to_use']
#     labels = ["rows_" + "_".join(map(str, row)) for row in rows_to_use]
#     param_sets = list(zip(rows_to_use, labels))

#     # Run the simulation for this replicate
#     results = Parallel(n_jobs=run_config['number_parallel_processes'])(
#         delayed(process_param_set)(rows, label, run_config)
#         for rows, label in param_sets
#     )

#     for (rows, label), res in zip(param_sets, results):
#         print(f"[Rep {i}] Completed simulation for {label} (rows={rows}): {res}")

# #%%
# import copy
# from tqdm.auto import tqdm
# path_to_data = "/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/median_parameter_simulations/"

# base_config = {

    
#     'n_cells': 6000, #Number of cells before division (number of twin pairs)
#     'simulation_time_before_division': 1000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
#     'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
#     'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
#     "path_to_connectivity_matrix": f"{path_to_data}/figure_2_simulation_details/interaction_matrix_A_both_repress_B.txt", #path to the connectivity matrix specifying the GRN to simulate
#     "param_csv": f"{path_to_data}/figure_2_simulation_details/median_param.csv", #Path to the parameters for all genes and interaction terms
#     "rows_to_use": [[6,6]], #Rows in the parameter's csv file for each gene - the length should be equal to number of genes in the system
#     "output_folder": f"{path_to_data}/figure_3_simulations/A_rep_B_B_rep_A/", #Path to folder to store simulation 
#     "log_file": f"{path_to_data}/figure_2_simulation_details/fig_2_median_parameter_simulations_6000_cells.jsonl", #Path to the log file
#     "type": "A_rep_B_B_rep_A",  # Name of the network used -- will be in the filename
#     "number_parallel_processes": 1, #Number of parameters to be run in parallel
#     "number_of_cores_per_parameter": 9, #Number of cores to be used per parameter (number_parallel_processes * number_of_cores_per_parameter = number of cores in your computer)
#     "noisy_division": True
# }

# for i in tqdm(range(20)):
#     run_config = copy.deepcopy(base_config)
#     run_config["type"] = f"{run_config['type']}_rep_{i}"

#     os.makedirs(run_config['output_folder'], exist_ok=True)
    
#     rows_to_use = run_config['rows_to_use']
#     labels = ["rows_" + "_".join(map(str, row)) for row in rows_to_use]
#     param_sets = list(zip(rows_to_use, labels))

#     # Run the simulation for this replicate
#     results = Parallel(n_jobs=run_config['number_parallel_processes'])(
#         delayed(process_param_set)(rows, label, run_config)
#         for rows, label in param_sets
#     )

#     for (rows, label), res in zip(param_sets, results):
#         print(f"[Rep {i}] Completed simulation for {label} (rows={rows}): {res}")

# #%%
# path_to_data = "/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/median_parameter_simulations/"

# base_config = {
#     'n_cells': 6000, #Number of cells before division (number of twin pairs)
#     'simulation_time_before_division': 1000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
#     'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
#     'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
#     "path_to_connectivity_matrix": f"{path_to_data}/figure_2_simulation_details/interaction_matrix_A_to_B.txt", #path to the connectivity matrix specifying the GRN to simulate
#     "param_csv": f"{path_to_data}/figure_2_simulation_details/median_param.csv", #Path to the parameters for all genes and interaction terms
#     "rows_to_use": [[0,0]], #Rows in the parameter's csv file for each gene - the length should be equal to number of genes in the system
#     "output_folder": f"{path_to_data}/noisy_division/A_to_B_50_percent/", #Path to folder to store simulation 
#     "log_file": f"{path_to_data}/figure_2_simulation_details/fig_2_median_parameter_simulations_noisy_division_cells.jsonl", #Path to the log file
#     "type": "A_to_B_50_percent",  # Name of the network used -- will be in the filename
#     "number_parallel_processes": 1, #Number of parameters to be run in parallel
#     "number_of_cores_per_parameter": 24, #Number of cores to be used per parameter (number_parallel_processes * number_of_cores_per_parameter = number of cores in your computer)
#     "noisy_division": True
# }

# for i in tqdm(np.arange(6,20)):
#     run_config = copy.deepcopy(base_config)
#     run_config["type"] = f"{run_config['type']}_rep_{i}"

#     os.makedirs(run_config['output_folder'], exist_ok=True)
    
#     rows_to_use = run_config['rows_to_use']
#     labels = ["rows_" + "_".join(map(str, row)) for row in rows_to_use]
#     param_sets = list(zip(rows_to_use, labels))

#     # Run the simulation for this replicate
#     results = Parallel(n_jobs=run_config['number_parallel_processes'])(
#         delayed(process_param_set)(rows, label, run_config)
#         for rows, label in param_sets
#     )

#     for (rows, label), res in zip(param_sets, results):
#         print(f"[Rep {i}] Completed simulation for {label} (rows={rows}): {res}")

# path_to_data = "/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/median_parameter_simulations/"

# base_config = {
#     'n_cells': 4000, #Number of cells before division (number of twin pairs)
#     'simulation_time_before_division': 2000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
#     'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
#     'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
#     "path_to_connectivity_matrix": f"{path_to_data}/figure_2_simulation_details/interaction_matrix_A_to_B.txt", #path to the connectivity matrix specifying the GRN to simulate
#     "param_csv": f"{path_to_data}/figure_2_simulation_details/median_param.csv", #Path to the parameters for all genes and interaction terms
#     "rows_to_use": [[3,3]], #Rows in the parameter's csv file for each gene - the length should be equal to number of genes in the system
#     "output_folder": f"{path_to_data}/figure_2_simulations_8000_cells/A_to_B_high_k_on/", #Path to folder to store simulation 
#     "log_file": f"{path_to_data}/figure_2_simulation_details/fig_2_median_parameter_simulations_8000_cells.jsonl", #Path to the log file
#     "type": "A_to_B_high_k_on",  # Name of the network used -- will be in the filename
#     "number_parallel_processes": 1, #Number of parameters to be run in parallel
#     "number_of_cores_per_parameter": 24, #Number of cores to be used per parameter (number_parallel_processes * number_of_cores_per_parameter = number of cores in your computer)
# }

# for i in tqdm(range(20)):
#     run_config = copy.deepcopy(base_config)
#     run_config["type"] = f"{run_config['type']}_rep_{i}"

#     os.makedirs(run_config['output_folder'], exist_ok=True)
    
#     rows_to_use = run_config['rows_to_use']
#     labels = ["rows_" + "_".join(map(str, row)) for row in rows_to_use]
#     param_sets = list(zip(rows_to_use, labels))

#     # Run the simulation for this replicate
#     results = Parallel(n_jobs=run_config['number_parallel_processes'])(
#         delayed(process_param_set)(rows, label, run_config)
#         for rows, label in param_sets
#     )

#     for (rows, label), res in zip(param_sets, results):
#         print(f"[Rep {i}] Completed simulation for {label} (rows={rows}): {res}")
# #%%

# import copy
# from tqdm.auto import tqdm

# path_to_data = "/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/median_parameter_simulations/"

# base_config = {
#     'n_cells': 4000, #Number of cells before division (number of twin pairs)
#     'simulation_time_before_division': 2000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
#     'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
#     'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
#     "path_to_connectivity_matrix": f"{path_to_data}/figure_2_simulation_details/interaction_matrix_A_to_B.txt", #path to the connectivity matrix specifying the GRN to simulate
#     "param_csv": f"{path_to_data}/figure_2_simulation_details/median_param.csv", #Path to the parameters for all genes and interaction terms
#     "rows_to_use": [[2,2]], #Rows in the parameter's csv file for each gene - the length should be equal to number of genes in the system
#     "output_folder": f"{path_to_data}/figure_2_simulations_8000_cells/A_to_B_low_k_on/", #Path to folder to store simulation 
#     "log_file": f"{path_to_data}/figure_2_simulation_details/fig_2_median_parameter_simulations_8000_cells.jsonl", #Path to the log file
#     "type": "A_to_B_low_k_on",  # Name of the network used -- will be in the filename
#     "number_parallel_processes": 1, #Number of parameters to be run in parallel
#     "number_of_cores_per_parameter": 24, #Number of cores to be used per parameter (number_parallel_processes * number_of_cores_per_parameter = number of cores in your computer)
# }

# for i in tqdm(np.arange(10,20)):
#     run_config = copy.deepcopy(base_config)
#     run_config["type"] = f"{run_config['type']}_rep_{i}"

#     os.makedirs(run_config['output_folder'], exist_ok=True)
    
#     rows_to_use = run_config['rows_to_use']
#     labels = ["rows_" + "_".join(map(str, row)) for row in rows_to_use]
#     param_sets = list(zip(rows_to_use, labels))

#     # Run the simulation for this replicate
#     results = Parallel(n_jobs=run_config['number_parallel_processes'])(
#         delayed(process_param_set)(rows, label, run_config)
#         for rows, label in param_sets
#     )

#     for (rows, label), res in zip(param_sets, results):
#         print(f"[Rep {i}] Completed simulation for {label} (rows={rows}): {res}")
