# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

# %%
#Script to create graph and reactions from input matrix
# %%
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

state_vars_per_gene = ["is_bursting", "unspliced_mRNA", "spliced_mRNA", "protein",
                      "k_on_adjusted", "total_mRNA", 
                      "mRNA_ever_produced", "protein_ever_produced"]
#Define indices for each state var
state_var_indices = {var: i for i, var in enumerate(state_vars_per_gene)}

# %%
#To read the input matrix
def read_input_matrix(path_to_matrix):
    """
    Reads the input matrix from the specified file path and counts number of genes.
    
    Parameters:
    path_to_matrix (str): The file path to the input matrix.
    
    Returns:
    np.ndarray: The input matrix as a NumPy array.
    """
    matrix = np.loadtxt(path_to_matrix, dtype='i', delimiter=',')
    dim_matrix = matrix.shape[0]
    return dim_matrix, matrix

# %%
#Read the parameters
def assign_parameters_to_genes(csv_path, rows=None, n_random=3):
    """
    Select specific rows or random rows from parameter CSV and map to genes.
    
    Args:
        csv_path (str): Path to CSV file
        rows (list, optional): Specific row indices to select. If None, selects random rows
        n_random (int): Number of random rows if rows=None
    
    Returns:
        tuple: (param_dict, row_mapping)
            param_dict: {gene_id: {param_name: value}}
            row_mapping: {gene_id: row_number}
    """
    df = pd.read_csv(csv_path, index_col=0)

    # If no specific rows are provided, select random rows to assugn to genes
    if rows is None:
        rows = np.random.choice(df.index, size=min(n_random, len(df)), replace=True)
    
    param_dict = {}
    row_mapping = {}
    
    for i, row in enumerate(rows):
        if row not in df.index:
            raise ValueError(f"Row {row} not found in the DataFrame.")
        gene_id = f"gene_{i+1}"
        param_dict[gene_id] = df.loc[row].to_dict()
        row_mapping[gene_id] = row
    # Create a matrix where genes are rows and parameters are columns
    param_matrix = pd.DataFrame.from_dict(param_dict, orient='index')

    #Add additional calculations for degradation, transcription rate, splicing rate, etc.
    param_matrix['mRNA_degradation_rate'] = np.log(2)/param_matrix['mrna_half_life']
    param_matrix['protein_degradation_rate'] = np.log(2)/param_matrix['protein_half_life']
    param_matrix['transcription_rate'] = param_matrix['burst_size'] * param_matrix['k_off']

    #Remove old columns that are not needed
    param_matrix.drop(columns=['mrna_half_life', 'protein_half_life', 'burst_size'], inplace=True, errors='ignore')

    return param_matrix, row_mapping

# %%
def save_complete_snapshot(state_array, state_var_indices, timestep=None):
    """
    Reshape from (n_genes, n_state_vars, n_cells) to (n_cells, n_genes×n_state_vars)
    """
    n_genes, n_state_vars, n_cells = state_array.shape
    
    # Create column names: gene_1_protein, gene_1_mRNA, gene_2_protein, etc.
    columns = []
    data = []
    
    for gene_idx in range(n_genes):
        for var_name, var_idx in state_var_indices.items():
            col_name = f"gene_{gene_idx+1}_{var_name}"
            columns.append(col_name)
            # Extract: state_array[gene, var, :] -> all cells for this gene-variable
            data.append(state_array[gene_idx, var_idx, :])
    
    # Stack columns: (n_genes×n_state_vars, n_cells) -> (n_cells, n_genes×n_state_vars)
    df_data = np.column_stack(data)
    
    # Create DataFrame: rows=cells, columns=gene_X_variable
    df = pd.DataFrame(df_data, columns=columns)
    df['cell_id'] = range(n_cells)
    
    if timestep is not None:
        df['timestep'] = timestep
    return df

# %%
def get_new_state(state_array, param_matrix, interaction_matrix, state_var_indices, 
                                   EC50_matrix=None, global_params=None, t=1/60):
    """
    Pure 3D matrix operations - everything happens simultaneously across all genes and cells.
    
    Args:
        state_array: shape (n_genes, n_state_vars, num_cells)
        param_matrix: DataFrame with genes as rows, parameters as columns
        interaction_matrix: shape (n_genes, n_genes) - regulatory interactions
        state_var_indices: dict mapping state variable names to indices
        global_params: dict of global parameters
        t: time step
    
    Returns:
        np.ndarray: updated state_array
    """

    if global_params is None:
        global_params = {"splicing_rate": np.log(2)/(7/60), "max_effect": 16, "n": 2}
    
    n_genes, n_state_vars, num_cells = state_array.shape
    new_state = state_array.copy()
    
    # Short aliases
    s = state_var_indices
    
    # Global parameters
    splicing_rate = global_params.get("splicing_rate", np.log(2)/(7/60))
    max_effect = global_params.get("max_effect", 16)
    n = global_params.get("n", 2)
    
    # === STEP 1: DEGRADATION - ALL GENES & CELLS AT ONCE ===
    
    # Create degradation rate arrays from DataFrame columns
    mrna_deg_rates = param_matrix['mRNA_degradation_rate'].values.reshape(n_genes, 1, 1)
    protein_deg_rates = param_matrix['protein_degradation_rate'].values.reshape(n_genes, 1, 1)
    
    # Apply degradation to entire state array slices - checked the dimensions
    unspliced_degraded = np.random.poisson(
        mrna_deg_rates * new_state[:, s["unspliced_mRNA"]:s["unspliced_mRNA"]+1, :] * t
    )
    spliced_degraded = np.random.poisson(
        mrna_deg_rates * new_state[:, s["spliced_mRNA"]:s["spliced_mRNA"]+1, :] * t
    )
    protein_degraded = np.random.poisson(
        protein_deg_rates * new_state[:, s["protein"]:s["protein"]+1, :] * t
    )
    
    # Update entire slices at once
    new_state[:, s["unspliced_mRNA"], :] = np.maximum(
        new_state[:, s["unspliced_mRNA"], :] - unspliced_degraded.squeeze(1), 0
    )
    new_state[:, s["spliced_mRNA"], :] = np.maximum(
        new_state[:, s["spliced_mRNA"], :] - spliced_degraded.squeeze(1), 0
    )
    new_state[:, s["protein"], :] = np.maximum(
        new_state[:, s["protein"], :] - protein_degraded.squeeze(1), 0
    )
    
    # === STEP 2: TRANSCRIPTION - ALL GENES & CELLS AT ONCE ===
    
    # Transcription rates from DataFrame column
    txn_rates = param_matrix['transcription_rate'].values.reshape(n_genes, 1)
    
    # Generate new mRNA for ALL genes and cells simultaneously
    new_mRNA = txn_rates * new_state[:, s["is_bursting"], :] * t
    
    # Add to unspliced mRNA pool - entire matrix operation
    new_state[:, s["unspliced_mRNA"], :] += new_mRNA
    new_state[:, s["mRNA_ever_produced"], :] += new_mRNA
    
    # === STEP 3: SPLICING - ALL GENES & CELLS AT ONCE ===
    
    # Splicing: unspliced → spliced for entire matrix
    spliced_amount = splicing_rate * new_state[:, s["unspliced_mRNA"], :] * t
    
    new_state[:, s["unspliced_mRNA"], :] -= spliced_amount
    new_state[:, s["spliced_mRNA"], :] += spliced_amount
    
    # === STEP 4: TRANSLATION - ALL GENES & CELLS AT ONCE ===
    
    # Protein production rates from DataFrame column
    protein_prod_rates = param_matrix['protein_production_rate'].values.reshape(n_genes, 1)
    
    # Create protein for ALL genes and cells simultaneously
    new_protein = protein_prod_rates * new_state[:, s["spliced_mRNA"], :] * t
    
    new_state[:, s["protein"], :] += new_protein
    new_state[:, s["protein_ever_produced"], :] += new_protein
    
    # Update total mRNA - entire matrix operation
    new_state[:, s["total_mRNA"], :] = (new_state[:, s["unspliced_mRNA"], :] + 
                                        new_state[:, s["spliced_mRNA"], :])
    
    # === STEP 5: K_ON ADJUSTMENT ===
    
    if interaction_matrix is not None and np.any(interaction_matrix != 0):
        # Get current protein levels: shape (n_genes, num_cells)
        protein_levels = new_state[:, s["protein"], :]
        
        if EC50_matrix is not None:
            # Use gene-specific EC50 values from the matrix
            # Create 3D arrays for vectorized Hill function calculation
            # protein_levels: (n_genes, num_cells) -> (1, n_genes, num_cells)
            # EC50_matrix: (n_genes, n_genes) -> (n_genes, n_genes, 1)
            
            protein_3d = protein_levels[np.newaxis, :, :]  # (1, n_genes, num_cells)
            EC50_3d = EC50_matrix[:, :, np.newaxis]        # (n_genes, n_genes, 1)
            
            # Hill function for all regulator-target-cell combinations
            # Shape: (n_genes, n_genes, num_cells)
            hill_responses = (protein_3d**n) / (EC50_3d**n + protein_3d**n)
            
            # Apply interaction strengths element-wise
            # interaction_matrix: (n_genes, n_genes) -> (n_genes, n_genes, 1)
            interaction_3d = interaction_matrix.T[:, :, np.newaxis]  # Transpose interaction_matrix
            weighted_responses = hill_responses * interaction_3d
            
            # Sum regulatory effects for each target gene across all regulators
            # Sum along regulator axis (axis=0): (n_genes, n_genes, num_cells) -> (n_genes, num_cells)
            regulatory_effects = np.sum(weighted_responses, axis=0)
        
        # Base k_on from DataFrame column
        base_k_on = param_matrix['k_on'].values.reshape(n_genes, 1)
        
        # Calculate adjusted k_on: shape (n_genes, num_cells)
        k_on_adjusted = base_k_on * (1 + regulatory_effects)
    else:
        # No regulation - use base k_on from DataFrame
        k_on_adjusted = np.tile(
            param_matrix['k_on'].values.reshape(n_genes, 1), (1, num_cells)
        )
    
    # Store in state array
    new_state[:, s["k_on_adjusted"], :] = k_on_adjusted
    
    # === STEP 6: BURST SWITCHING - ALL GENES & CELLS AT ONCE ===
    
    # Current burst states
    is_bursting = new_state[:, s["is_bursting"], :].astype(bool)
    
    # k_off rates from DataFrame column
    k_off_rates = param_matrix['k_off'].values.reshape(n_genes, 1)
    
    # Switch probabilities for ALL genes and cells simultaneously
    switch_off_prob = np.random.exponential(1 / k_off_rates, size=(n_genes, num_cells)) < t
    switch_on_prob = np.random.exponential(1 / k_on_adjusted) < t
    
    # Apply switching logic - pure boolean operations on entire matrices
    should_switch_off = is_bursting & switch_off_prob
    should_switch_on = (~is_bursting) & switch_on_prob
    
    # Update burst states for entire matrix
    new_burst_state = (is_bursting | should_switch_on) & (~should_switch_off)
    new_state[:, s["is_bursting"], :] = new_burst_state.astype(float)
    
    # === FINAL: ENSURE NON-NEGATIVE VALUES ===
    new_state = np.maximum(new_state, 0)
    
    return new_state

def select_parameter_rows(csv_path, rows=None, n_random=3):
    """
    Select parameter rows and return matrix format.
    
    Args:
        csv_path (str): Path to CSV file
        rows (list, optional): Specific row indices to select
        n_random (int): Number of random rows if rows=None
    
    Returns:
        tuple: (param_matrix, row_mapping)
    """
    df = pd.read_csv(csv_path, index_col=0)
    
    if rows is None:
        rows = np.random.choice(df.index, size=min(n_random, len(df)), replace=False)
    
    param_dict = {}
    row_mapping = {}
    
    for i, row in enumerate(rows):
        if row not in df.index:
            raise ValueError(f"Row {row} not found in the DataFrame.")
        gene_id = f"gene_{i+1}"
        param_dict[gene_id] = df.loc[row].to_dict()
        row_mapping[gene_id] = row
    
    # Create parameter matrix: genes as rows, parameters as columns
    param_matrix = pd.DataFrame.from_dict(param_dict, orient='index')
    
    return param_matrix, row_mapping

def save_complete_snapshot(state_array, state_var_indices, timestep=None):
    """
    Convert state array to DataFrame for saving.
    
    Args:
        state_array: shape (n_genes, n_state_vars, num_cells)
        state_var_indices: dict mapping state names to indices
        timestep: optional timestep label
        
    Returns:
        DataFrame: rows=cells, columns=gene_X_variable
    """
    n_genes, n_state_vars, num_cells = state_array.shape
    
    # Create column names and data
    columns = []
    data = []
    
    for gene_idx in range(n_genes):
        for var_name, var_idx in state_var_indices.items():
            col_name = f"gene_{gene_idx+1}_{var_name}"
            columns.append(col_name)
            data.append(state_array[gene_idx, var_idx, :])
    
    # Create DataFrame
    df_data = np.column_stack(data)
    df = pd.DataFrame(df_data, columns=columns)
    
    if timestep is not None:
        df['timestep'] = timestep
    df['cell_id'] = range(num_cells)
    
    return df

def get_steady_state_protein_levels(param_matrix, interaction_matrix, global_params=None, target_hill=0.5):
    """
    Calculate steady-state protein levels for all genes to use as EC50 values.
    
    Args:
        param_matrix: DataFrame with genes as rows, parameters as columns
        interaction_matrix: shape (n_genes, n_genes) - regulatory interactions
        global_params: dict of global parameters
        target_hill: target Hill function value at steady state (default 0.5)
    
    Returns:
        np.ndarray: steady-state protein levels for each gene (shape: n_genes,)
    """

    
    if global_params is None:
        global_params = {"splicing_rate": np.log(2)/(7/60), "max_effect": 16, "n": 2}
    
    n_genes = param_matrix.shape[0]
    
    # Access parameters by column name
    k_on_values = param_matrix['k_on'].to_numpy()
    k_off_values = param_matrix['k_off'].to_numpy()
    txn_rate_values = param_matrix['transcription_rate'].to_numpy()
    mrna_deg_values = param_matrix['mRNA_degradation_rate'].to_numpy()
    protein_prod_values = param_matrix['protein_production_rate'].to_numpy()
    protein_deg_values = param_matrix['protein_degradation_rate'].to_numpy()
    
    splicing_rate = global_params.get("splicing_rate", np.log(2)/(7/60))
    max_effect = global_params.get("max_effect", 16.0)
    
    def steady_state_system(protein_levels):
        """
        System of equations for steady-state protein levels.
        Each gene's steady state depends on its regulators.
        """
        equations = []
        
        for gene_idx in range(n_genes):
            # Get regulators for this gene
            regulators = np.where(interaction_matrix[:, gene_idx] != 0)[0]

            if len(regulators) == 0:
                # No regulation - simple steady state
                b_gene = k_on_values[gene_idx] / (k_on_values[gene_idx] + k_off_values[gene_idx])
            else:
                # Calculate effective k_on based on regulators
                regulatory_effect = 0
                for reg_idx in regulators:
                    reg_protein = protein_levels[reg_idx]
                    interaction_strength = interaction_matrix[reg_idx, gene_idx] #Is +1 or -1.
                    
                    # Hill function at target value (e.g., 0.5)
                    # Solve: target_hill = protein^n / (EC50^n + protein^n)
                    # If we want hill=0.5 when protein=reg_protein, then EC50=reg_protein
                    hill_response = target_hill  # Fixed at target value
                    regulatory_effect += hill_response * interaction_strength
                
                # Effective k_on
                k_on_eff = np.maximum(k_on_values[gene_idx] * max_effect * (1 + regulatory_effect), 0)
                b_gene = k_on_eff / (k_on_eff + k_off_values[gene_idx])
            
            # Steady-state calculation through the cascade
            # Transcription -> Unspliced -> Spliced -> Protein
            u_ss = (txn_rate_values[gene_idx] * b_gene) / (mrna_deg_values[gene_idx] + splicing_rate)
            s_ss = (u_ss * splicing_rate) / mrna_deg_values[gene_idx]
            p_predicted = (s_ss * protein_prod_values[gene_idx]) / protein_deg_values[gene_idx]
            
            # Equation: predicted protein level should equal input protein level
            equations.append(p_predicted - protein_levels[gene_idx])
        
        return equations
    
    # Initial guess - simple steady states without regulation
    initial_guess = []
    for gene_idx in range(n_genes):
        b_simple = k_on_values[gene_idx] / (k_on_values[gene_idx] + k_off_values[gene_idx])
        u_simple = (txn_rate_values[gene_idx] * b_simple) / (mrna_deg_values[gene_idx] + splicing_rate)
        s_simple = (u_simple * splicing_rate) / mrna_deg_values[gene_idx]
        p_simple = (s_simple * protein_prod_values[gene_idx]) / protein_deg_values[gene_idx]
        initial_guess.append(p_simple)
    
    # Solve the system of equations
    try:
        steady_state_proteins = fsolve(steady_state_system, initial_guess)
        
        # Ensure positive values
        steady_state_proteins = np.maximum(steady_state_proteins, 0.1)
        
        return steady_state_proteins
    
    except Exception as e:
        print(f"Failed to solve steady state: {e}")
        return np.array(initial_guess)

def create_EC50_matrix(param_matrix, interaction_matrix, global_params=None):
    """
    Create EC50 matrix where EC50[i,j] = steady-state protein level of regulator j
    when it regulates target i.
    
    Args:
        param_matrix: DataFrame with parameter values
        interaction_matrix: regulatory interaction matrix
        global_params: global parameters
    
    Returns:
        np.ndarray: EC50 matrix (shape: n_genes, n_genes)
    """
    # Get steady-state protein levels
    steady_state_proteins = get_steady_state_protein_levels(
        param_matrix, interaction_matrix, global_params
    )
    
    n_genes = len(steady_state_proteins)
    EC50_matrix = np.zeros((n_genes, n_genes))
    
    # For each regulatory interaction, set EC50 to regulator's steady-state level
    for target in range(n_genes):
        for regulator in range(n_genes):
            if interaction_matrix[regulator, target] != 0:
                EC50_matrix[regulator, target] = steady_state_proteins[regulator]
    
    return EC50_matrix
# %%
def create_random_cells(n_genes, num_cells=1000, mean_protein_levels=None):
    """
    Create a random state array for a given number of genes and cells.
    
    Args:
        n_genes (int): Number of genes
        num_cells (int): Number of cells
    
    Returns:
        np.ndarray: Random state array of shape (n_genes, n_state_vars, num_cells)
    """
    state_vars_per_gene = ["is_bursting", "unspliced_mRNA", "spliced_mRNA", "protein",
                          "k_on_adjusted", "total_mRNA", 
                          "mRNA_ever_produced", "protein_ever_produced"]
    
    state_var_indices = {var: i for i, var in enumerate(state_vars_per_gene)}
    
    # Initialize state array with zeros
    state_array = np.zeros((n_genes, len(state_vars_per_gene), num_cells))
    #Protein levels for each gene are between 0 and 3*mean_protein_levels for the gene
    if mean_protein_levels is None:
        mean_protein_levels = np.random.uniform(0, 100, n_genes)
    else:
        mean_protein_levels = np.diag(mean_protein_levels)
    # Randomly initialize protein levels
    state_array[:, state_var_indices["protein"], :] = np.random.uniform(0, 3 * mean_protein_levels[:, np.newaxis], size=(n_genes, num_cells))
    return state_array, state_var_indices

# %%
def simulate(path_to_matrix, parameter_sheet_path, num_cells, global_params= None, rows=None):
    
    if global_params is None:
        # Default global parameters if not provided
        global_params = {
            "splicing_rate": np.log(2)/(7/60),  # Splicing rate in per minute
            "max_effect": 16,  # Maximum effect for Hill function
            "n": 2,  # Hill coefficient
        }
    print("Reading input matrix and parameters...")
    n_genes, interaction_matrix = read_input_matrix(path_to_matrix)
    param_matrix, row_mapping = assign_parameters_to_genes(parameter_sheet_path, rows=rows, n_random=n_genes)

    #Calculate EC50 matrix
    print("Calculating EC50 matrix...")
    EC50_matrix = create_EC50_matrix(param_matrix, interaction_matrix)

    #Burn in - not needed because I already have mean and burn-in is just giving me one more mean

    #Create random cells
    print("Creating random cells...")
    state_array, state_var_indices = create_random_cells(n_genes, num_cells=num_cells, mean_protein_levels=EC50_matrix)

    #simulation time settings
    t = 1/60  # 1 minute time step
    total_time = 13 * 24 * 60 
    time_to_start_measurement = 12 * 24 * 60  # Start measuring after 12 days
    frequency_measurement = 60  # Measure every 60 minutes\
    steps = np.arange(0, total_time, int(t*60))
    division_time = 12 * 24 * 60  # Division occurs after 12 days
    pre_division_steps = np.arange(0, total_time, division_time)
    after_division_steps = steps[steps > division_time] - division_time
    state_list = []
    replicate_list = []
    print("Starting simulation...")
    for curr_step in pre_division_steps:
        state_array = get_new_state(state_array.copy(), param_matrix, interaction_matrix, state_var_indices, 
                                   EC50_matrix, global_params, t=t)
    
    print("Simulation before division complete.")
    replicate_state = state_array.copy()
    state_list.append(save_complete_snapshot(state_array, state_var_indices, timestep=curr_step))
    state_list.append(save_complete_snapshot(replicate_state, state_var_indices, timestep=curr_step))

    for curr_step in after_division_steps:
        state_array = get_new_state(state_array.copy(), param_matrix, interaction_matrix, state_var_indices, 
                                   EC50_matrix, global_params, t=t)
        replicate_state = get_new_state(replicate_state.copy(), param_matrix, interaction_matrix, state_var_indices, 
                                   EC50_matrix, global_params, t=t)
        
        if curr_step >= time_to_start_measurement and curr_step % frequency_measurement == 0:
            state_list.append(save_complete_snapshot(state_array, state_var_indices, timestep=curr_step))
            replicate_list.append(save_complete_snapshot(replicate_state, state_var_indices, timestep=curr_step))
    
    print("Simulation after division complete.")
    state_array_df = pd.concat(state_list, ignore_index=True)
    state_array_df['replicate'] = 0
    replicate_array_df = pd.concat(replicate_list, ignore_index=True)
    replicate_array_df['replicate'] = 1
    # Combine both dataframes
    combined_df = pd.concat([state_array_df, replicate_array_df], ignore_index=True)
    # Save the combined DataFrame to a CSV file
    return combined_df



