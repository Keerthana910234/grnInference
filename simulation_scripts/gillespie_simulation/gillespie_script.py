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

def generate_reaction_network_from_matrix(interaction_matrix: np.ndarray):
    """
    Generate a reaction network from a given interaction matrix.

    This function constructs a reaction network based on gene interactions defined 
    in the input interaction matrix. It generates reactions for activation, 
    regulation, inactivation, mRNA production/degradation, and protein production/degradation 
    for each gene in the network.

    Args:
        interaction_matrix (np.ndarray): A square matrix representing gene interactions. 
            Each element interaction_matrix[i, j] indicates the regulatory effect of gene i 
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
            - gene_list (List[str]): A list of gene names generated from the interaction matrix.

    Notes:
        - The propensity functions for reactions are defined using a set of predefined templates.
        - Parameters for each reaction are dynamically generated based on the gene and interaction 
          matrix information.
        - The function aggregates reactions with identical species and changes into a single row 
          with combined propensity functions.
    """
    n_genes = interaction_matrix.shape[0]
    gene_list = [f"gene_{i+1}" for i in range(n_genes)]
    prop = {
        "regulatory": "(({sign}*{p_add})*({activator}_protein**{n})/({k}**{n}+{activator}_protein**{n}))*{target}_I",
        "activation": "{p_on}*{target}_I",
        "inactivation": "{p_off}*{target}_A",
        "mRNA_prod": "{p_prod_mRNA}*{target}_A",
        "mRNA_deg": "{p_deg_mRNA}*{target}_mRNA",
        "protein_prod": "{p_prod_protein}*{target}_mRNA",
        "protein_deg": "{p_deg_protein}*{target}_protein"
    }
    reactions = []
    for j, target in enumerate(gene_list):
        param = lambda p: f"{{{p}_{target}}}"
        # activation
        expr = prop["activation"].replace("{p_on}", param("p_on")).replace("{target}", target)
        reactions.append({"species1":f"{target}_A","change1":1,
                          "species2":f"{target}_I","change2":-1,
                          "propensity":expr,"time":"-"})
        # regulation
        regulators = np.where(interaction_matrix[:,j]!=0)[0]
        for i in regulators:
            source = gene_list[i]
            sign = int(np.sign(interaction_matrix[i,j]))
            edge = f"{source}_to_{target}"
            expr = prop["regulatory"]\
                .replace("{sign}",str(sign))\
                .replace("{p_add}",f"{{p_add_{edge}}}")\
                .replace("{n}",f"{{n_{edge}}}")\
                .replace("{k}",f"{{k_{edge}}}")\
                .replace("{activator}",source)\
                .replace("{target}",target)
            reactions.append({"species1":f"{target}_A","change1":1,
                              "species2":f"{target}_I","change2":-1,
                              "propensity":expr,"time":"-"})
        # inactivation
        expr = prop["inactivation"].replace("{p_off}",param("p_off")).replace("{target}",target)
        reactions.append({"species1":f"{target}_I","change1":1,
                          "species2":f"{target}_A","change2":-1,
                          "propensity":expr,"time":"-"})
        # production/degradation
        for label,suffix,chg in [
            ("mRNA_prod","mRNA",1),("mRNA_deg","mRNA",-1),
            ("protein_prod","protein",1),("protein_deg","protein",-1)
        ]:
            expr = prop[label].replace("{target}",target)
            for p in ["p_prod_mRNA","p_deg_mRNA","p_prod_protein","p_deg_protein"]:
                expr = expr.replace(f"{{{p}}}",param(p))
            reactions.append({"species1":f"{target}_{suffix}","change1":chg,
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
        tuple: A tuple containing:
            - param_dict (dict): A dictionary mapping parameter names (formatted 
                                 as "{parameter_gene}") to their values.
            - param_matrix (pd.DataFrame): A DataFrame where rows correspond to 
                                           genes and columns correspond to parameter values.
    """
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        raise ValueError(f"Parameter csv file not found at path: {csv_path}")
    n = len(gene_list)
    if rows is None:
        rows = np.random.choice(df.index, size=n, replace=True)
    param_dict = {}
    param_matrix = {}
    for i,row in enumerate(rows):
        gene = gene_list[i]
        if int(row) in df.index:
            vals = df.loc[int(row)].copy()
        else:
            raise KeyError(f"Row index {int(row)} not found in the DataFrame.")
        vals["p_deg_mRNA"] = np.log(2)/vals["mrna_half_life"]
        vals["p_deg_protein"] = np.log(2)/vals["protein_half_life"]
        vals.drop(["mrna_half_life","protein_half_life"],axis=0,inplace=True,errors="ignore")
        param_matrix[gene] = vals
        for k,v in vals.items():
            param_dict[f"{{{k}_{gene}}}"] = float(v)
    return param_dict, pd.DataFrame(param_matrix).T

def generate_k_from_steady_state_calc(param_dict, interaction_matrix, gene_list,
                                      target_hill=0.5, scale_k=None):
    """
    Calculate steady-state protein levels and assign rate constants (k values) 
    for gene interactions based on the provided parameters and interaction 

    Args:
        param_dict (dict): Dictionary containing parameters for gene regulation, 
            including burst probabilities, production rates, degradation rates, 
            and interaction strengths.
        interaction_matrix (numpy.ndarray): Matrix representing gene interactions, 
            where non-zero values indicate regulatory relationships and their signs 
            (positive for activation, negative for repression).
        gene_list (list): List of gene names corresponding to the rows and columns 
            of the interaction matrix.
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
        - Regulatory effects are computed using the interaction matrix and scaled 
          by the target Hill coefficient (default is 0.5).
        - Rate constants (k values) are assigned based on steady-state protein 
          levels and multiplied by the scaling matrix.
    """
    n_genes = len(gene_list)
    if scale_k is None:
        scale_k = np.ones((n_genes, n_genes))
    protein_levels = np.zeros(n_genes)
    for i,gene in enumerate(gene_list):
        p_on = param_dict[f'{{p_on_{gene}}}']
        p_off = param_dict[f'{{p_off_{gene}}}']
        p_prod_mRNA = param_dict[f'{{p_prod_mRNA_{gene}}}']
        p_deg_mRNA  = param_dict[f'{{p_deg_mRNA_{gene}}}']
        p_prod_prot = param_dict[f'{{p_prod_protein_{gene}}}']
        p_deg_prot  = param_dict[f'{{p_deg_protein_{gene}}}']
        reg_eff = 0.0
        regs = np.where(interaction_matrix[:,i]!=0)[0]
        for r in regs:
            edge = f"{gene_list[r]}_to_{gene}"
            p_add = param_dict.get(f"{{p_add_{edge}}}", 0.0)
            sign = interaction_matrix[r,i]
            reg_eff += target_hill * p_add * sign
        p_on_eff = p_on + reg_eff
        burst_prob = p_on_eff/(p_on_eff+p_off)
        m = p_prod_mRNA * burst_prob / p_deg_mRNA
        protein_levels[i] = max(m * p_prod_prot / p_deg_prot, 0.1)
    # assign k values
    for i,src in enumerate(gene_list):
        for j,tgt in enumerate(gene_list):
            if interaction_matrix[i,j]!=0:
                key = f"{{k_{src}_to_{tgt}}}"
                param_dict[key] = protein_levels[i]*scale_k[i,j]
    return protein_levels, param_dict

def add_interaction_terms(param_dict, interaction_matrix, gene_list,
                          n_matrix=None, p_add_matrix=None):
    """
    Adds interaction terms to the parameter dictionary based on the interaction matrix 
    and gene list, and calculates steady-state paramet
    Parameters:
        param_dict (dict): Dictionary to store the interaction parameters.
        interaction_matrix (numpy.ndarray): Matrix representing interactions between genes.
                                            Non-zero values indicate an interaction.
        gene_list (list): List of gene names corresponding to the rows and columns of 
                          the interaction matrix.
        n_matrix (numpy.ndarray, optional): Matrix specifying the 'n' parameter for each 
                                            interaction. Defaults to a matrix filled with 2.0.
        p_add_matrix (numpy.ndarray, optional): Matrix specifying the 'p_add' parameter for 
                                                each interaction. Defaults to a matrix filled 
                                                with 1
    Returns:
        dict: Updated parameter dictionary with interaction terms added.
    """
    n = len(gene_list)
    if n_matrix is None:
        n_matrix = np.full((n,n),2.0)
    if p_add_matrix is None:
        p_add_matrix = np.full((n,n),10.0)
    for i in range(n):
        for j in range(n):
            if interaction_matrix[i,j]!=0:
                edge = f"{gene_list[i]}_to_{gene_list[j]}"
                param_dict[f"{{n_{edge}}}"]     = float(n_matrix[i,j])
                param_dict[f"{{p_add_{edge}}}"] = float(p_add_matrix[i,j])
    return generate_k_from_steady_state_calc(param_dict, interaction_matrix, gene_list)

def setup_gillespie_params_from_reactions(init_states: pd.DataFrame,
                                          reactions: pd.DataFrame,
                                          param_dictionary: dict):
    """
    Sets up the parameters required for Gillespie simulation based on initial states, reaction definitions, 
    and a parameter dictionary. This function generates the initial population, update matrix, 
    and a compiled function for updating prope
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

# %% SSA core JIT
@numba.njit(fastmath=True)
def sample_discrete(probs):
    """
    Samples an index from a discrete probability distribution.

    This function takes an array of probabilities and returns an index
    sampled according to the given probabilities. The probabilities should
    sum to 1.

    Args:
        probs (numpy.ndarray): A 1D array of probabilities representing the 
            discrete probability distribution. Each element in the array 
            corresponds to the probability of selecting the respective index.

    Returns:
        int: The index sampled based on the given probabilities.
    """
    q = np.random.rand()
    cum = 0.0
    for i in range(probs.shape[0]):
        cum += probs[i]
        if cum >= q:
            return i
    return probs.shape[0]-1

@numba.njit(fastmath=True)
def gillespie_draw(prop_func, prop, pop, t):
    
    """
    Calculates the event that has to occur and the time it takes.

    This function determines the next event in a stochastic simulation based on the Gillespie algorithm. It uses the provided propensity function to calculate the propensities, selects an event based on the cumulative distribution of propensities, and computes the time until the next event.

    Args:
        prop_func (function): A function that calculates the propensities given the current state.
        prop (numpy.ndarray): Array of propensities for each possible event.
        pop (numpy.ndarray): Current population state.
        t (float): Current simulation time.
    Returns:
        tuple:
            - int: Index of the event to occur (-1 if no event occurs).
            - float: Time until the next event (-1.0 if no event occurs).
    """
    prop_func(prop, pop, t)
    total = 0.0
    for r in range(prop.shape[0]):
        total += prop[r]
    if total <= 0:
        return -1, -1.0
    dt = np.random.exponential(1.0 / total)
    q = np.random.rand()
    cum = 0.0
    for r in range(prop.shape[0]):
        cum += prop[r] / total
        if cum >= q:
            return r, dt
    return prop.shape[0] - 1, dt

# %% Vectorized extraction

def extract_mrna_protein_fast(samples: np.ndarray, species_index: dict,
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
def simulate_cells_numba(update_propensities, update_matrix, pop0_mat, time_points, verbose_flags):
    """
    Simulates the dynamics of multiple cells using the Gillespie algorithm.

    Parameters:
    -----------
    update_propensities : callable
        A function that updates the reaction propensities for a given cell population and time.
        It should accept three arguments: propensities array, population array, and current time.
    update_matrix : numpy.ndarray
        A 2D array of shape (n_rxns, n_species) representing the stoichiometric matrix for reactions.
        Each row corresponds to a reaction, and each column corresponds to a species.
    pop0_mat : numpy.ndarray
        A 2D array of shape (n_species, n_cells) representing the initial population of species for each cell.
        Each column corresponds to a cell, and each row corresponds to a species.
    time_points : numpy.ndarray
        A 1D array of time points at which the population samples are recorded.
    verbose_flags : numpy.ndarray
        A 1D array of flags for each cell to indicate if the simulation encountered issues (e.g., stuck state).

    Returns:
    --------
    samples : numpy.ndarray
        A 3D array of shape (n_cells, n_time, n_species) containing the simulated population of species
        for each cell at each time point. The dimensions correspond to cells, time points, and species.

    Notes:
    ------
    - The Gillespie algorithm is used to simulate stochastic dynamics of chemical reactions.
    - If a cell's simulation gets stuck (e.g., due to zero total propensity), it will attempt to recover
      up to `max_attempts` times before marking the cell as problematic in `verbose_flags`.
    - The simulation fills skipped time points with the previous population state to ensure continuity.
    """
    n_species, n_cells = pop0_mat.shape
    n_time = time_points.shape[0]
    n_rxns = update_matrix.shape[0]
    samples = np.zeros((n_cells, n_time, n_species), dtype=np.int64)

    for cell in prange(n_cells):
        pop = pop0_mat[:, cell].copy()
        prev = pop.copy()
        t = time_points[0]
        samples[cell, 0, :] = pop
        i_time = 1
        stuck_counter = 0
        max_attempts = 10000
        prop = np.zeros(n_rxns, dtype=np.float64)

        while i_time < n_time:
            update_propensities(prop, pop, t)
            total = prop.sum()
            if total <= 0:
                stuck_counter += 1
                if stuck_counter > max_attempts:
                    verbose_flags[cell] = 1
                    break
                continue
            stuck_counter = 0
            dt = np.random.exponential(1.0 / total)
            q = np.random.rand()
            cum = 0.0
            for i in range(n_rxns):
                cum += prop[i] / total
                if cum >= q:
                    rxn = i
                    break
            else:
                rxn = n_rxns - 1

            prev = pop.copy()
            for s in range(n_species):
                pop[s] += update_matrix[rxn, s]
            t += dt

            # Fill all previous time points when no event occurred
            while i_time < n_time and t >= time_points[i_time]:
                samples[cell, i_time, :] = prev
                i_time += 1
    return samples

# %%
# Check for steady state
def is_steady_state(samples, time_points, mean_tol=0.05, std_tol=0.05,
                    slope_tol=0.05, window_frac=0.25, verbose=False):
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

    if verbose:
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
        - The function uses a JIT-compiled helper function `simulate_cells_numba` for efficient simulation.
        - Warnings are printed for cells that encounter issues during simulation:
            - Cell stuck due to zero propensities for too long.
    """
    n_species = pop0.shape[0]
    pop0_mat = np.tile(pop0[:, None], (1, n_cells))
    verbose_flags = np.zeros(n_cells, dtype=np.int64)
    print("Starting simulation with {} cells \n".format(n_cells))
    samples = simulate_cells_numba(update_propensities, update_matrix, pop0_mat, time_points, verbose_flags)
    for cell in range(n_cells):
        if verbose_flags[cell] == 1:
            print(f"⚠️ WARNING: Cell {cell} got stuck (zero propensities).")
    return samples

# --- Extraction ---
def extract_mrna_protein_fast(samples, species_index, types=('mRNA','protein')):
    n_cells, n_time, _ = samples.shape
    sel = [(name,idx) for name,idx in species_index.items() if any(name.endswith(t) for t in types)]
    names, idxs = zip(*sel)
    data = samples[:,:,idxs].reshape(n_cells*n_time, len(idxs))
    cell_ids   = np.repeat(np.arange(n_cells), n_time)
    time_steps = np.tile(np.arange(n_time), n_cells)
    df = pd.DataFrame(data, columns=names)
    df.insert(0,'time_step',time_steps)
    df.insert(0,'cell_id',  cell_ids)
    return df

# --- Worker for a single parameter set ---
def process_param_set(rows, label, base_config):
    # base_config contains common parameters: paths, p_add_matrix, n_matrix, time_points
    set_num_threads(8)
    print(f"[Worker {label}] Using {get_num_threads()} threads for rows={rows}\n")
    # Unpack base_config
    path_to_matrix = base_config['path_to_matrix']
    param_csv      = base_config['param_csv']
    # p_add_matrix   = base_config['p_add_matrix']
    # n_matrix       = base_config['n_matrix']
    time_points    = base_config['time_points']
    n_cells        = base_config['n_cells']

    # Build reactions and parameters for this row set
    n_genes, mat = read_input_matrix(path_to_matrix)
    reactions_df, gene_list = generate_reaction_network_from_matrix(mat)
    init_states = generate_initial_state_from_genes(gene_list)
    param_dict, _ = assign_parameters_to_genes(param_csv, gene_list, rows)
    n_matrix = np.zeros((n_genes, n_genes))
    p_add_matrix = np.zeros((n_genes, n_genes))
    for i in range(n_genes):
        for j in range(n_genes):
            #Check in the interaction matrix if the edge is a regulation ot not
            if mat[i, j] != 0:
                edge = f"{gene_list[i]}_to_{gene_list[j]}"
                n_matrix[i,j]     = param_dict.get(f"{{n_{edge}}}", 2.0)
                p_add_matrix[i,j] = param_dict.get(f"{{p_add_{edge}}}", 10.0)

    steady_state, full_param_dict = add_interaction_terms(param_dict, mat, gene_list,
                                                          n_matrix=n_matrix,
                                                          p_add_matrix=p_add_matrix)

    pop0, update_matrix, update_prop, species_index = setup_gillespie_params_from_reactions(
        init_states, reactions_df, full_param_dict)

    # 1) Run base simulation
    base_samples = run_simulation(update_prop, update_matrix, pop0, time_points, n_cells)
    if not is_steady_state(base_samples, time_points):
        print(f"⚠️ Base simulation (basal) for {label} not steady.")
    
    # 2) Replicate into two to create daughter cells
    final_states = base_samples[:, -1, :]
    rep_time = np.arange(0, 48, 1)
    pop0_rep = np.concatenate([final_states.T, final_states.T], axis=1)
    rep_samples = simulate_cells_numba(update_prop, update_matrix, pop0_rep, rep_time, np.zeros(2*n_cells, dtype=np.int64))
    
    # 3) Extract from simulation and label
    df_rep = extract_mrna_protein_fast(rep_samples, species_index)
    n_total = 2 * n_cells
    replicate_ids = np.repeat([1, 2], n_cells)
    clone_ids = np.tile(np.arange(n_cells), 2)

    df_rep['replicate'] = replicate_ids[df_rep['cell_id']]
    df_rep['clone_id'] = clone_ids[df_rep['cell_id']]
    df_rep['cell_id'] = df_rep.index // len(rep_time)  # optional: restore per-cell unique index
    
    # 4) Save
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    id = uuid.uuid4().hex[:8]
    prefix = f"{label}_{timestamp}_ncells_{n_cells}_{base_config['type']}_{id}"
    df_rep.to_csv(f"{base_config['output_folder']}/df_{prefix}.csv", index=False)
    np.savetxt(f"{base_config['output_folder']}/samples_{prefix}.csv", rep_samples.reshape(2*n_cells, -1), delimiter=",")
    record = {
        "id": id,
        "rows": rows,
        "n_cells": n_cells,
        "timestamp": timestamp,
        "param_dict": full_param_dict,
        "steady_state": steady_state.tolist()
    }
    os.makedirs(os.path.dirname(base_config['log_file']), exist_ok=True)
    with open(base_config['log_file'],"a") as f:
        f.write(json.dumps(record) + "\n")
    return prefix

#%%
# --- Main execution with parallel parameter sets ---
if __name__ == "__main__":
    # Base configuration - the commented out lines can be used instead of providing arguments to the file (e.g. if using it as ipynb notebook)
    base_config = {
        'time_points':    np.arange(0, 800, 1), #Time to reach steady state
        'n_cells':        10000 #Before division
        # "path_to_matrix":  "/home/mzo5929/Keerthana/grnInference/simulation_data/general_simulation_data/test_data/matrix101.txt",
        # "param_csv":      "/home/mzo5929/Keerthana/grnInference/simulation_data/gillespie_simulation/sim_details/lhc_sampled_parameters_positive_reg.csv",
        # "row_to_start":      0,
        # "output_folder":      "/path/to/save/saimulation/output",
        # "log_file":      "/path/to/log.jsonl",
        # "type":      "A_to_B",
        # 
    }
    # Define 4 parameter sets (rows) and labels
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Gillespie simulation with specified inputs.")
    parser.add_argument("--matrix_path", type=str, required=True, help="Path to the interaction matrix file.")
    parser.add_argument("--param_csv", type=str, required=True, help="Path to the parameter CSV file.")
    parser.add_argument("--row_to_start", type=int, required=True, help="Row of parameter file to start.")
    parser.add_argument("--output_folder", type=str , required=True, help="Folder to save the simulation.")
    parser.add_argument("--log_file", type=str , required=True, help="Json file to save log.")
    parser.add_argument("--type", type=str , required=True, help="Type of regulation.")
    args = parser.parse_args()

    # Update base configuration with parsed arguments
    base_config["path_to_matrix"] = args.matrix_path
    base_config["param_csv"] = args.param_csv
    base_config["row_to_start"] = int(args.row_to_start)
    base_config["output_folder"] = args.output_folder
    base_config["log_file"] = args.log_file
    base_config["type"] = args.type

    df = pd.read_csv(base_config['param_csv'])
    #This simulation will run the rows from the row_to_start till the end (useful to batch across multiple runs)
    row_list = [[i, i+1] for i in range(base_config["row_to_start"], len(df), 2)]
    labels = [f"row_{i}_{i+1}" for i in range( base_config["row_to_start"], len(df), 2)]
    param_sets = list(zip(row_list, labels))
    # Use 32 cores split into 4 workers (8 threads each)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_param_set, rows, label, base_config)
                   for rows, label in param_sets]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Param sets"):  
            prefix = fut.result()
            print(f"Completed simulation: {prefix}")
# %%
