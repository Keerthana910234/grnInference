#Updates
# # Optimized Gillespie-SSA Simulation Pipeline
# %% Input utilities
import os
import uuid
import json
from datetime import datetime
import re
import numpy as np
import pandas as pd
import numba
from numba import prange, set_num_threads, get_num_threads
from tqdm.auto import tqdm
import time
import concurrent.futures
import argparse
import gc 

# %% Input utilities

def read_input_matrix(path_to_matrix: str) -> (int, np.ndarray):
    """
    Reads an input matrix from a specified file path and returns its dimensions and content.

    Args:
        path_to_matrix (str): The file path to the matrix file. The file should contain
                              a comma-separated matrix of integers.

    Returns:
        tuple: A tuple containing:
            - int: The number of rows in the matrix.
            - np.ndarray: The matrix as a NumPy array. If the matrix is a single value,
                          it is reshaped into a 1x1 array.

    Raises:
        ValueError: If the file cannot be loaded.
    """
    try:
        matrix = np.loadtxt(path_to_matrix, dtype=int, delimiter=',')
        if matrix.ndim == 0:
            matrix = matrix.reshape((1,1))
        return matrix.shape[0], matrix
    except Exception as e:
        raise ValueError(f"Error loading matrix from {path_to_matrix}: {e}")


def assign_parameters_to_genes(csv_path, gene_list, rows=None):
    """
    Assigns parameters to a list of genes based on values from a CSV file.

    This function reads a CSV file containing parameter values, selects rows 
    either randomly or based on the provided indices, and assigns the parameters 
    to the specified genes. It calculates additional parameters such as 
    degradation rates for mRNA and protein based on their respective half-lives.

    Args:
        csv_path (str): Path to the CSV file containing parameter values. 
                        The file should have columns including 'mrna_half_life' 
                        and 'protein_half_life'.
        gene_list (list): List of gene names to which parameters will be assigned.
        rows (list, optional): List of row indices to select from the CSV file. 
                               If None, rows are randomly selected with replacement. 
                               Defaults to None.

    Returns:
            param_dict (dict): A dictionary mapping parameter names (formatted 
                                 as "{parameter_gene}") to their values.
    """
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        raise ValueError(f"Parameter csv file not found at path: {csv_path}")
    n = len(gene_list)
    
    param_dict = {}
    for i,row in enumerate(rows):
        gene = gene_list[i]
        if int(row) in df.index:
            vals = df.loc[int(row)].copy()
        else:
            raise KeyError(f"Row index {int(row)} not found in the DataFrame.")
        vals["k_deg_mRNA"] = np.log(2)/vals["mrna_half_life"]
        vals["k_deg_protein"] = np.log(2)/vals["protein_half_life"]
        vals.drop(["mrna_half_life","protein_half_life"],axis=0,inplace=True,errors="ignore")
        for k, v in vals.items():
            if "_to_" in k:
                param_dict[f"{{{k}}}"] = float(v)  # Interaction parameter: keep as is
            else:
                param_dict[f"{{{k}_{gene}}}"] = float(v)  # Gene-specific parameter
    return param_dict


def generate_reaction_network_from_matrix(connectivity_matrix: np.ndarray):
    """
    Generate a reaction network from a given connectivity matrix.

    This function constructs a reaction network based on gene interactions defined 
    in the input connectivity matrix. It generates reactions for gene activation/inactivation, 
    regulation, mRNA production/degradation, and protein production/degradation 
    for each gene in the network.

    Args:
        connectivity_matrix (np.ndarray): A square matrix representing gene interactions. 
            Each element connectivity_matrix[i, j] indicates the regulatory effect of gene i 
            on gene j. Positive values represent activation, negative values represent 
            repression, and zero indicates no interaction.

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - reactions_df (pd.DataFrame): A DataFrame containing the reaction network. 
              Each row represents a reaction with the following columns:
                - 'species1': The species involved in the reaction.
                - 'change1': The change in the count of 'species1'.
                - 'species2': The second species involved in the reaction (if applicable).
                - 'change2': The change in the count of 'species2'.
                - 'time': Placeholder for reaction time (currently set to "-").
                - 'propensity': The propensity function for the reaction.
            - gene_list (List[str]): A list of gene names generated from the connectivity matrix.

    Notes:
        - The propensity functions for reactions are defined using a set of predefined templates.
        - Parameters for each reaction are dynamically generated based on the gene and interaction 
          matrix information.
        - The function aggregates reactions with identical species and changes into a single row 
          with combined propensity functions.
    """
    n_genes = connectivity_matrix.shape[0]
    gene_list = [f"gene_{i+1}" for i in range(n_genes)]
    prop = {
        "regulatory": "(({sign}*{k_add})*({tf}_protein**{n})/({k}**{n}+{tf}_protein**{n}))*{curr_gene}_I",
        "activation": "{k_on}*{curr_gene}_I",
        "inactivation": "{k_off}*{curr_gene}_A",
        "mRNA_prod": "{k_prod_mRNA}*{curr_gene}_A",
        "mRNA_deg": "{k_deg_mRNA}*{curr_gene}_mRNA",
        "protein_prod": "{k_prod_protein}*{curr_gene}_mRNA",
        "protein_deg": "{k_deg_protein}*{curr_gene}_protein"
    }

    reactions = []
    
    for j, curr_gene in enumerate(gene_list):
        param = lambda k: f"{{{k}_{curr_gene}}}"
        # activation
        expr = prop["activation"].replace("{k_on}", param("k_on")).replace("{curr_gene}", curr_gene)
        reactions.append({"species1":f"{curr_gene}_A","change1":1,
                          "species2":f"{curr_gene}_I","change2":-1,
                          "propensity":expr,"time":"-"})
        # regulation
        regulators_index = np.where(connectivity_matrix[:,j]!=0)[0]
        for i in regulators_index:
            regulator = gene_list[i]
            sign = int(np.sign(connectivity_matrix[i,j]))
            edge = f"{regulator}_to_{curr_gene}"
            expr = prop["regulatory"]\
                .replace("{sign}",str(sign))\
                .replace("{k_add}",f"{{k_add_{edge}}}")\
                .replace("{n}",f"{{n_{edge}}}")\
                .replace("{k}",f"{{k_{edge}}}")\
                .replace("{tf}",regulator)\
                .replace("{curr_gene}",curr_gene)
            reactions.append({"species1":f"{curr_gene}_A","change1":1,
                              "species2":f"{curr_gene}_I","change2":-1,
                              "propensity":expr,"time":"-"})
        # inactivation
        expr = prop["inactivation"].replace("{k_off}",param("k_off")).replace("{curr_gene}",curr_gene)
        reactions.append({"species1":f"{curr_gene}_I","change1":1,
                          "species2":f"{curr_gene}_A","change2":-1,
                          "propensity":expr,"time":"-"})
        # production/degradation
        for label,suffix,chg in [
            ("mRNA_prod","mRNA",1),("mRNA_deg","mRNA",-1),
            ("protein_prod","protein",1),("protein_deg","protein",-1)
        ]:
            expr = prop[label].replace("{curr_gene}",curr_gene)
            for k in ["k_prod_mRNA","k_deg_mRNA","k_prod_protein","k_deg_protein"]:
                expr = expr.replace(f"{{{k}}}",param(k))
            reactions.append({"species1":f"{curr_gene}_{suffix}","change1":chg,
                              "species2":"-","change2":"-",
                              "propensity":expr,"time":"-"})
    df = pd.DataFrame(reactions)
    df['propensity'] = df['propensity'].astype(str)
    reactions_df = (
        df.groupby(['species1','change1','species2','change2','time'])['propensity']
          .agg(lambda x: ' + '.join(x)).reset_index()
    )
    return reactions_df, gene_list

def generate_initial_state_from_genes(gene_list):
    """
    Generate the initial state for a list of genes.

    This function creates a DataFrame representing the initial state of species
    associated with each gene in the provided list. For each gene, the following
    species are initialized:
    - `<gene>_A`: Active state, initialized with a count of 0.
    - `<gene>_I`: Inactive state, initialized with a count of 1.
    - `<gene>_mRNA`: Messenger RNA, initialized with a count of 0.
    - `<gene>_protein`: Protein, initialized with a count of 0.

    Args:
        gene_list (list of str): A list of gene names for which the initial states
                                 are to be generated.

    Returns:
        pandas.DataFrame: A DataFrame containing the initial states of the species
                          for each gene. Each row represents a species with its
                          name (`species`) and initial count (`count`).
    """
    states = []
    for g in gene_list:
        states += [
            {"species":f"{g}_A","count":0},
            {"species":f"{g}_I","count":1},
            {"species":f"{g}_mRNA","count":0},
            {"species":f"{g}_protein","count":0},
        ]
    return pd.DataFrame(states)

def generate_k_from_steady_state_calc(param_dict, connectivity_matrix, gene_list,
                                      target_hill=0.5, scale_k=None):
    """
    Calculate steady-state protein levels and assign rate constants (k values) 
    for gene interactions based on the provided parameters and interaction 

    Args:
        param_dict (dict): Dictionary containing parameters for gene regulation, 
            including burst probabilities, production rates, degradation rates, 
            and interaction strengths.
        connectivity_matrix (numpy.ndarray): Matrix representing gene interactions, 
            where non-zero values indicate regulatory relationships and their signs 
            (positive for activation, negative for repression).
        gene_list (list): List of gene names corresponding to the rows and columns 
            of the connectivity matrix.
        target_hill (float, optional): Hill coefficient used to scale regulatory 
            effects. Default is 0.5.
        scale_k (numpy.ndarray, optional): Scaling matrix for rate constants. If 
            None, defaults to a matrix of ones with the same dimensions as the 
            interaction 
    Returns:
        tuple: A tuple containing:
            - protein_levels (numpy.ndarray): Array of steady-state protein levels 
              for each gene.
            - param_dict (dict): Updated dictionary with assigned rate constants 
              (k values) for gene intera
    Notes:
        - The function calculates steady-state protein levels based on burst 
          probabilities and production/degradation rates.
        - Regulatory effects are computed using the connectivity matrix and scaled 
          by the target Hill coefficient (default is 0.5).
        - Rate constants (k values) are assigned based on steady-state protein 
          levels and multiplied by the scaling matrix.
    """
    n_genes = len(gene_list)
    if scale_k is None:
        scale_k = np.ones((n_genes, n_genes))
    protein_levels = np.zeros(n_genes)
    for i,gene in enumerate(gene_list):
        k_on = param_dict[f'{{k_on_{gene}}}']
        k_off = param_dict[f'{{k_off_{gene}}}']
        k_prod_mRNA = param_dict[f'{{k_prod_mRNA_{gene}}}']
        k_deg_mRNA  = param_dict[f'{{k_deg_mRNA_{gene}}}']
        k_prod_prot = param_dict[f'{{k_prod_protein_{gene}}}']
        k_deg_prot  = param_dict[f'{{k_deg_protein_{gene}}}']
        reg_eff = 0.0
        regs = np.where(connectivity_matrix[:,i]!=0)[0]
        for r in regs:
            edge = f"{gene_list[r]}_to_{gene}"
            k_add = param_dict.get(f"{{k_add_{edge}}}", 0.0)
            sign = connectivity_matrix[r,i]
            reg_eff += target_hill * k_add * sign
        k_on_eff = k_on + reg_eff
        burst_prob = k_on_eff/(k_on_eff+k_off)
        m = k_prod_mRNA * burst_prob / k_deg_mRNA
        protein_levels[i] = max(m * k_prod_prot / k_deg_prot, 0.1)
    # assign k values
    for i, src in enumerate(gene_list):
        for j, tgt in enumerate(gene_list):
            if connectivity_matrix[i,j]!=0:
                key = f"{{k_{src}_to_{tgt}}}"
                param_dict[key] = protein_levels[i]*scale_k[i,j]
    print(param_dict)
    return protein_levels, param_dict

def add_interaction_terms(param_dict, connectivity_matrix, gene_list,
                          n_matrix=None, k_add_matrix=None):
    """
    Adds interaction terms to the parameter dictionary based on the connectivity matrix 
    and gene list, and calculates steady-state paramet
    Parameters:
        param_dict (dict): Dictionary to store the interaction parameters.
        connectivity_matrix (numpy.ndarray): Matrix representing interactions between genes.
                                            Non-zero values indicate an interaction.
        gene_list (list): List of gene names corresponding to the rows and columns of 
                          the connectivity matrix.
        n_matrix (numpy.ndarray, optional): Matrix specifying the 'n' parameter for each 
                                            interaction. Defaults to a matrix filled with 2.0.
        k_add_matrix (numpy.ndarray, optional): Matrix specifying the 'k_add' parameter for 
                                                each interaction. Defaults to a matrix filled 
                                                with 1
    Returns:
        dict: Updated parameter dictionary with interaction terms added.
    """
    n = len(gene_list)
    if n_matrix is None:
        n_matrix = np.full((n,n),2.0)
    if k_add_matrix is None:
        k_add_matrix = np.full((n,n),6.0)
    for i in range(n):
        for j in range(n):
            if connectivity_matrix[i,j]!=0:
                edge = f"{gene_list[i]}_to_{gene_list[j]}"
                param_dict[f"{{n_{edge}}}"]     = float(n_matrix[i,j])
                param_dict[f"{{k_add_{edge}}}"] = float(k_add_matrix[i,j])
    return generate_k_from_steady_state_calc(param_dict, connectivity_matrix, gene_list)

def setup_gillespie_params_from_reactions(init_states: pd.DataFrame,
                                          reactions: pd.DataFrame,
                                          param_dictionary: dict):
    """
    Sets up the parameters required for Gillespie simulation based on initial states, reaction definitions, 
    and a parameter dictionary. This function generates the initial population, update matrix, 
    and a compiled function for updating propensities
    Args:
        init_states (pd.DataFrame): A DataFrame containing the initial states of species. 
                                    Must include columns 'species' and 'count'.
        reactions (pd.DataFrame): A DataFrame defining the reactions. 
                                  Must include columns 'species1', 'species2', 'change1', 'change2', and 'propensity'.
        param_dictionary (dict): A dictionary mapping parameter names to their values, 
                                 used for substituting placeholders in propensity f
    Returns:
        tuple: A tuple containing:
            - pop0 (np.ndarray): Initial population counts as a NumPy array of integers.
            - update_matrix (np.ndarray): A matrix defining the changes in species counts for each reaction.
            - update_propensities (function): A compiled function for updating propensities using numba.
            - species_index (dict): A dictionary mapping species names to their 
    Raises:
        ValueError: If any placeholders in the propensity formulas are missing from the parameter dic
    Notes:
        - The function dynamically generates and compiles a propensity update function using numba for performance.
        - Species names and parameters in the propensity formulas are replaced with their respective indices and values.
    """
    species_index = {s:i for i,s in enumerate(init_states['species'])}
    pop0 = init_states['count'].values.astype(np.int64)
    update_matrix = []
    prop_formulas = []
    missing = []
    for i,row in reactions.iterrows():
        delta = [0]*len(species_index)
        a1,a2 = row['species1'], row['species2']
        delta[species_index[a1]] = int(row['change1'])
        if a2!='-':
            delta[species_index[a2]] = int(row['change2'])
        update_matrix.append(delta)
        expr = row['propensity']
        # inject species
        for s,idx in species_index.items():
            expr = expr.replace(s, f"pop[idx_{s}]")
        # inject params
        placeholders = set(re.findall(r"{[^}]+}", expr))
        miss = placeholders - set(param_dictionary.keys())
        if miss:
            missing.append((i, miss))
            continue
        for k,v in param_dictionary.items():
            expr = expr.replace(k, str(v))
        line = f"prop[{i}] = {expr}"
        prop_formulas.append(line)
    if missing:
        raise ValueError(f"Missing params in propensities: {missing}")
    # build update function
    src = ["@numba.njit(fastmath=True)",
           "def update_propensities(prop, pop, t):"]
    for s,i in species_index.items():
        src.append(f"    idx_{s} = {i}")
    for L in prop_formulas:
        src.append("    " + L)
    ns = "\n".join(src)
    loc = {}
    exec(ns, {'numba':numba}, loc)
    return pop0, np.array(update_matrix, dtype=np.int64), loc['update_propensities'], species_index

# %% Vectorized extraction

def convert_samples_to_df(samples: np.ndarray, species_index: dict,
                              types=('mRNA','protein')) -> pd.DataFrame:
    """
    Extracts mRNA and protein data from simulation samples and organizes it into a pandas DataF
    Parameters:
        samples (np.ndarray): A 3D numpy array of shape (n_cells, n_time, n_species) containing simulation data.
                              Each entry represents the count of a species at a given cell and time step.
        species_index (dict): A dictionary mapping species names to their respective indices in the samples array.
        types (tuple, optional): A tuple of strings specifying the types of species to extract (e.g., 'mRNA', 'protein').
                                 Defaults to ('mRNA', 'prote
    Returns:
        pd.DataFrame: A pandas DataFrame containing the extracted data. The DataFrame includes the following columns:
                      - 'cell_id': The ID of the cell (integer).
                      - 'time_step': The time step (integer).
                      - Columns for each extracted species, named according to the species_index keys.
    """
    n_cells, n_time, _ = samples.shape
    sel = [(name,idx) for name,idx in species_index.items()
           if any(name.endswith(t) for t in types)]
    names, idxs = zip(*sel)
    data = samples[:,:,idxs].reshape(n_cells*n_time, len(idxs))
    cell_ids   = np.repeat(np.arange(n_cells), n_time)
    time_steps = np.tile(np.arange(n_time), n_cells)
    df = pd.DataFrame(data, columns=names)
    df.insert(0,'time_step',time_steps)
    df.insert(0,'cell_id',cell_ids)
    return df

@numba.njit(parallel=True, fastmath=True)
def gillespie_simulation_all_cells(update_propensities, update_matrix, pop0_mat, time_points, verbose_flags):
    n_species, n_cells = pop0_mat.shape
    n_time = time_points.shape[0] #Number of time points to sample
    n_rxns = update_matrix.shape[0]
    samples = np.empty((n_cells, n_time, n_species), dtype=np.int64)

    for cell in prange(n_cells):
        pop = pop0_mat[:, cell].copy()
        t = time_points[0]
        i_time = 0 #This is the index of current time point in the time_points array (starts at 0)
        stuck_counter = 0
        max_attempts = 10000
        prop = np.zeros(n_rxns, dtype=np.float64)
        next_tp = time_points[0] #This is the next time point where sample will be measured

        while i_time < n_time:
            update_propensities(prop, pop, t)
            total = prop.sum()

            if total <= 0:  # no events possible
                stuck_counter += 1
                if stuck_counter > max_attempts:
                    verbose_flags[cell] = 1
                    samples[cell, i_time:, :] = pop
                    break
                samples[cell, i_time, :] = pop
                # i_time += 1
                continue

            stuck_counter = 0
            reaction_time = np.random.exponential(1.0 / total)
            t += reaction_time


            while i_time < n_time and t >= next_tp: #Check if we crossed a sampling time - save as needed
                samples[cell, i_time, :] = pop
                i_time += 1
                if i_time < n_time: #updating next time point only if simulation total time is not over
                    next_tp = time_points[i_time]

            # Vectorized reaction selection
            cum_props = np.cumsum(prop)
            r = np.searchsorted(cum_props, np.random.rand() * total)
            pop += update_matrix[r]
    return samples


# %%
# Check for steady state
def is_steady_state(samples, time_points, mean_tol=0.05, std_tol=0.05,
                    slope_tol=0.05, window_frac=0.1, verbose=False):
    """
    Check if the simulation has reached steady state.

    Args:
        samples (np.ndarray): Array of shape (n_cells, n_time, n_species)
        time_points (np.ndarray): Array of time values
        mean_tol (float): Max relative change in mean allowed
        std_tol (float): Max relative change in std allowed
        slope_tol (float): Max absolute slope allowed
        window_frac (float): Fraction of final time used to assess steady state
        verbose (bool): Whether to print detailed output

    Returns:
        bool: True if steady state is reached
    """
    n_cells, n_time, n_species = samples.shape
    window = int(n_time * window_frac)
    if window < 2:
        raise ValueError("Window too small for steady state check.")

    data = samples[:, -window:, :]  # shape: (n_cells, window, n_species)
    mean_traj = data.mean(axis=0)   # shape: (window, n_species)
    std_traj  = data.std(axis=0)    # shape: (window, n_species)

    # Mean & std relative change over last window
    rel_mean_change = np.abs(mean_traj[-1] - mean_traj[0]) / (mean_traj[0] + 1e-6)
    rel_std_change  = np.abs(std_traj[-1] - std_traj[0]) / (std_traj[0] + 1e-6)

    max_mean_change = rel_mean_change.max()
    max_std_change  = rel_std_change.max()

    steady_mean_std = max_mean_change < mean_tol and max_std_change < std_tol

    # Slope check
    times = time_points[-window:]
    slopes = np.zeros(n_species)
    for g in range(n_species):
        y = mean_traj[:, g]
        x = times
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        slopes[g] = m

    max_abs_slope = np.abs(slopes).max()
    steady_slope = max_abs_slope < slope_tol

    is_steady = steady_mean_std or steady_slope

    print(f"🧪 Steady-state check:")
    print(f"  ➤ Max relative mean change: {max_mean_change:.4e}")
    print(f"  ➤ Max relative std  change: {max_std_change:.4e}")
    print(f"  ➤ Max abs slope:             {max_abs_slope:.4e}")
    print(f"  ➤ Steady by mean/std:        {steady_mean_std}")
    print(f"  ➤ Steady by slope:           {steady_slope}")
    print(f"  ➤ Final decision:            {is_steady}")

    return is_steady

# %% Wrapping functions 

def run_simulation(update_propensities, update_matrix, pop0, time_points, n_cells=1000):
    """
    Simulates the dynamics of a population of cells using the Gillespie algorithm.

    Parameters:
        update_propensities (callable): A function to compute the propensities for reactions.
        update_matrix (numpy.ndarray): The stoichiometry matrix defining the system's reactions.
        pop0 (numpy.ndarray): Initial population vector for all species (shape: [n_species]).
        time_points (numpy.ndarray): Array of time points at which to sample the population.
        n_cells (int, optional): Number of cells to simulate. Defaults to 1000.

    Returns:
        numpy.ndarray: A 3D array containing the simulated population data. 
                       Shape: [n_species, len(time_points), n_cells].

    Notes:
        - The function uses a JIT-compiled helper function `gillespie_simulation_all_cells` for efficient simulation.
        - Warnings are printed for cells that encounter issues during simulation:
            - Cell stuck due to zero propensities for too long.
    """
    n_species = pop0.shape[0]
    pop0_mat = np.tile(pop0[:, None], (1, n_cells))
    verbose_flags = np.zeros(n_cells, dtype=np.int64)
    samples = gillespie_simulation_all_cells(update_propensities, update_matrix, pop0_mat, time_points, verbose_flags)
    for cell in range(n_cells):
        if verbose_flags[cell] == 1:
            print(f"⚠️ WARNING: Cell {cell} got stuck (zero propensities).")
    return samples

# --- Worker for a single parameter set ---
def process_param_set(rows, label, base_config):
    # base_config contains common parameters: paths, k_add_matrix, n_matrix, time_points
    # set_num_threads(6)
    print(f"[Worker {label}] Using {get_num_threads()} threads for rows={rows}\n")
    # Unpack base_config
    path_to_connectivity_matrix = base_config['path_to_connectivity_matrix']
    param_csv      = base_config['param_csv']
    # k_add_matrix   = base_config['k_add_matrix']
    # n_matrix       = base_config['n_matrix']
    time_points    = np.arange(0, base_config['simulation_time_before_division'], 1)
    sample_twins_time_points    = np.arange(0, base_config['twin_simulation_time_after_division'] + base_config['twin_measurement_resolution'], base_config['twin_measurement_resolution']) 
    n_cells        = base_config['n_cells']
    # Build reactions and parameters for this row set
    n_genes, connectivity_matrix = read_input_matrix(path_to_connectivity_matrix)
    assert len(rows) >= n_genes, "The number of parameter rows entered is less than the number of genes"
    reactions_df, gene_list = generate_reaction_network_from_matrix(connectivity_matrix)
    init_states = generate_initial_state_from_genes(gene_list)
    param_dict = assign_parameters_to_genes(param_csv, gene_list, rows)
    n_matrix = np.zeros((n_genes, n_genes))
    k_add_matrix = np.zeros((n_genes, n_genes))
    for i in range(n_genes):
        for j in range(n_genes):
            #Check in the connectivity matrix if the edge is a regulation ot not
            if connectivity_matrix[i, j] != 0:
                edge = f"{gene_list[i]}_to_{gene_list[j]}"
                n_matrix[i,j]     = param_dict.get(f"{{n_{edge}}}", 2.0)
                k_add_matrix[i,j] = param_dict.get(f"{{k_add_{edge}}}", 6.0)
    print("Done until addition of interaction terms")
    steady_state, full_param_dict = add_interaction_terms(param_dict, connectivity_matrix, gene_list,
                                                          n_matrix=n_matrix,
                                                          k_add_matrix=k_add_matrix)

    pop0, update_matrix, update_prop, species_index = setup_gillespie_params_from_reactions(
        init_states, reactions_df, full_param_dict)
    print("Starting base simulation")
    # 1) Run base simulation
    base_samples = run_simulation(update_prop, update_matrix, pop0, time_points, n_cells)
    if not is_steady_state(base_samples, time_points):
        print(f"⚠️ Base simulation (basal) for {label} not steady.")
        # Log the issue in a separate file
        error_record = {
            "id": uuid.uuid4().hex[:8],
            "rows": rows,
            "timestamp": datetime.now().strftime("%d%m%Y_%H%M%S"),
            "issue": "Base simulation not steady",
            "label": label
        }
        log_folder = os.path.join(os.path.dirname(base_config['log_file']))
        os.makedirs(log_folder, exist_ok=True)
        log_file_path = os.path.join(log_folder, f"error_log.jsonl")
        with open(log_file_path, "a") as log_file:
            log_file.write(json.dumps(error_record) + "\n")
        
    df_base = convert_samples_to_df(base_samples, species_index)
    
    # 2) Replicate into two to create daughter cells
    final_states = base_samples[:, -1, :]
    del base_samples
    gc.collect()
    rep_time = sample_twins_time_points
    pop0_rep = np.concatenate([final_states.T, final_states.T], axis=1)
    rep_samples = gillespie_simulation_all_cells(update_prop, update_matrix, pop0_rep, rep_time, np.zeros(2*n_cells, dtype=np.int64))
    
    # 3) Extract from simulation and label
    df_rep = convert_samples_to_df(rep_samples, species_index)
    n_total = 2 * n_cells
    replicate_ids = np.repeat([1, 2], n_cells)
    clone_ids = np.tile(np.arange(n_cells), 2)

    df_rep['replicate'] = replicate_ids[df_rep['cell_id']]
    df_rep['clone_id'] = clone_ids[df_rep['cell_id']]
    df_rep['cell_id'] = df_rep.index // len(rep_time) 
    
    # 4) Save
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    id = uuid.uuid4().hex[:8]
    prefix = f"{label}_{timestamp}_ncells_{n_cells}_{base_config['type']}_{id}"
    df_rep.to_csv(f"{base_config['output_folder']}/df_{prefix}.csv", index=False)
    # np.savetxt(f"{base_config['output_folder']}/samples_{prefix}.csv", rep_samples.reshape(2*n_cells, -1), delimiter=",")
    df_base.to_csv(f"{base_config['output_folder']}/test_df_{prefix}.csv", index=False)
    record = {
        "id": id,
        "rows": rows,
        "n_cells": n_cells,
        "type": base_config['type'],
        "timestamp": timestamp,
        "param_dict": full_param_dict,
        "steady_state": steady_state.tolist()
    }
    os.makedirs(os.path.dirname(base_config['log_file']), exist_ok=True)
    with open(base_config['log_file'],"a") as f:
        f.write(json.dumps(record) + "\n")
    return f"{base_config['output_folder']}/df_{prefix}.csv"

#%%
# --- Main execution with parallel parameter sets ---
if __name__ == "__main__":
    set_num_threads(12)
    print("Threads Numba will use:", get_num_threads())
    root = "/projects/b1042/GoyalLab/Keerthana/"
    # Base configuration - the commented out lines can be used instead of providing arguments to the file (e.g. if using it as ipynb notebook)
    base_config = {
        'time_points':    np.arange(0, 2500, 1), #Time to reach steady state
        'n_cells':        10000, #Before division
        # "path_to_matrix":  "/path/to/interaction/matrix.txt",
        # "param_csv":      "/path/to/parameters.csv",
        # "row_to_start":      0,
        # "output_folder":      "/path/to/output/folder/",
        # "log_file":      "/path/to/log.jsonl",
        # "type":      "A_to_B",
        
    }

    import numpy as np
    root = "/home/mzo5929/Keerthana/grnInference/"

    base_config = {
        'n_cells': 1000, #Number of cells before division (number of twin pairs)
        'simulation_time_before_division': 1000, #The time used to run the initial cells before division. User must set this time to ensure the population reaches steady state [hours]
        'twin_simulation_time_after_division': 48, #The time twin cells are simulated after division and measurements are stored in the output[hours]
        'twin_measurement_resolution': 1, #The time between each measurement of twin cells [hours]. For example, if twin_sampling_duration is 12 and twin_measurement_resolution is 1, the final dataframe will contain hourly measurements for 12 hours (0 is birth).
        "path_to_connectivity_matrix": f"{root}/simulation_data/median_parameter_simulations/simulation_details/interaction_matrix_A_to_B_A_to_C.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"{root}/simulation_data/median_parameter_simulations/simulation_details/median_param_3_gene.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0, 1, 2]], #Rows in the parameter's csv file for each gene - the length should be equal to number of genes in the system
        "output_folder": f"{root}/simulation_data/median_parameter_simulations/new_simulation", #Path to folder to store simulation 
        "log_file": f"{root}/simulation_data/median_parameter_simulations/simulation_details/median_parameter_simulations.jsonl", #Path to the log file
        "type": "Fan_out",  # Name of the network used -- will be in the filename
        "number_of_parallel_parameters": 1, #Number of parameters to be run in parallel
        "number_of_cores_per_parameter": 8 #Number of cores to be used per parameter (number_of_parallel_parameters * number_of_cores_per_parameter = number of cores in your computer)
    }


    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Gillespie simulation with specified inputs.")
    parser.add_argument("--path_to_connectivity_matrix", type=str, required=True, help="Path to the connectivity matrix file specifying the GRN to simulate.")
    parser.add_argument("--param_csv", type=str, required=True, help="Path to the parameters for all genes and interaction terms.")
    parser.add_argument("--row_to_start", type=int, required=True, help="Row of parameter file to start for this batch of simulations.")
    parser.add_argument("--output_folder", type=str , required=True, help="Path to output folder to store simulation.")
    parser.add_argument("--log_file", type=str , required=True, help="Json file to save log.")
    parser.add_argument("--type", type=str , required=True, help="Name of the network used -- will be in the filename.")
    parser.add_argument("--number_parallel_processes", type=int, default=1, required=False, help="Number of parallel parameter sets to be run at once (default: 1).")
    parser.add_argument("--n_genes", type=int, default=2, required=False, help="Number of genes in the system (default: 2).")
    parser.add_argument("--n_cells", type=int, default=5000, required=False, help="Number of cells in the system (default: 5000).")
    parser.add_argument("--steady_state_time", type=int, default=2500, required=False, help="Number of hours to run to achieve steady state (default: 250h0).")
    parser.add_argument("--twin_sampling_duration", type=int, default=48, required=False, help="Number of hours to run after cell division for collecting twin data (default: 48h).")
    parser.add_argument("--twin_sampling_frequency", type=int, default=1, required=False, help="The time duration between every twin measurement (default: 1). For example, if it is 1h, then, data is stored eevry hour.")
    args = parser.parse_args()

    # # Update base configuration with parsed arguments
    base_config["path_to_connectivity_matrix"] = args.matrix_path
    base_config["param_csv"] = args.param_csv
    base_config["row_to_start"] = int(args.row_to_start)
    base_config["output_folder"] = args.output_folder
    base_config["log_file"] = args.log_file
    base_config["type"] = args.type
    base_config["number_parallel_processes"] = args.number_parallel_processes
    base_config["n_genes"] = args.n_genes
    base_config["n_cells"] = args.n_cells
    base_config["steady_state_time"] = args.steady_state_time
    base_config["twin_sampling_duration"] = args.twin_sampling_duration
    base_config["twin_sampling_frequency"] = args.twin_sampling_frequency

    os.makedirs(base_config["output_folder"], exist_ok = True)
    try:
        df = pd.read_csv(base_config['param_csv'])
    except FileNotFoundError:
        print(f"Error: The file {base_config['param_csv']} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

    # Read the connectivity matrix before using it
    path_to_connectivity_matrix = base_config["path_to_connectivity_matrix"]
    n_genes, mat = read_input_matrix(path_to_connectivity_matrix)  # Ensure mat is defined
    start_pair = base_config["row_to_start"]  # row_to_start now refers to pair_id
    end_pair = start_pair + 650
    print(f"start_pair: {start_pair}, end_pair: {end_pair}")
    row_list = []
    labels = []

    # Group by pair_id and collect rows for each group
    for pair in range(start_pair, end_pair + 1):
        subset = df[df["pair_id"] == pair].sort_values("gene_id")
        rows = subset.index.tolist()

        # Ensure only complete groups are taken
        if len(rows) >= n_genes:
            row_list.append(rows[:n_genes])
            labels.append(f"row_{'_'.join(map(str, rows))}")

    param_sets = list(zip(row_list, labels))
    print(len(param_sets))
    # Use 32 cores split into 4 workers (8 threads each)
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_param_set, rows, label, base_config)
                   for rows, label in param_sets]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Param sets"):  
            prefix = fut.result()
            print(f"Completed simulation: {prefix}")
