import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from correlation_analysis_helpers import dict_to_matrix

def steady_state_calc(param_dict, interaction_matrix, gene_list,
                                   sim_data, scale_k=None):
    """
    Calculates regulated steady-state protein levels using empirical Hill responses
    from simulation data. Assigns k values based on computed steady states.

    Args:
        param_dict (dict): Contains kinetic and interaction parameters.
        interaction_matrix (np.ndarray): Shape (n_genes, n_genes) — regulator → target.
        gene_list (list): Ordered list of gene names.
        sim_data (pd.DataFrame): Must contain 'gene_{i}_protein' for each gene.
        scale_k (np.ndarray): Optional scaling matrix for assigning k.

    Returns:
        protein_levels_sim_data (np.ndarray): Steady state protein levels estimated using simulation data.
    """
    def hill_fn(x, n, k):
        x = np.asarray(x)
        return x ** n / (x ** n + k ** n)

    n_genes = len(gene_list)
    if scale_k is None:
        scale_k = np.ones((n_genes, n_genes))

    protein_levels_sim_data = np.zeros(n_genes)

    for i, gene in enumerate(gene_list):
        p_on = param_dict[f'k_on_{gene}']
        p_off = param_dict[f'k_off_{gene}']
        p_prod_mRNA = param_dict[f'k_prod_mRNA_{gene}']
        p_deg_mRNA = param_dict[f'k_deg_mRNA_{gene}']
        p_prod_prot = param_dict[f'k_prod_protein_{gene}']
        p_deg_prot = param_dict[f'k_deg_protein_{gene}']

        reg_eff = 0.0
        regulators = np.where(interaction_matrix[:, i] != 0)[0]
        for r in regulators:
            src_gene = gene_list[r]
            edge = f"{src_gene}_to_{gene}"
            p_add = param_dict.get(f"k_add_{edge}", 0.0)
            n_val = param_dict.get(f"n_{edge}", 1.0)
            k_val = param_dict.get(f"k_{edge}", 1.0)
            sign = interaction_matrix[r, i]
            key = f"gene_{r+1}_protein"
            if key not in sim_data:
                raise ValueError(f"{key} not found in sim_data")

            x_vals = np.asarray(sim_data[key])
            hill_vals = hill_fn(x_vals, n_val, k_val)
            hill_response = np.mean(hill_vals)
            reg_eff += p_add * hill_response * sign

        p_on_eff = p_on + reg_eff
        burst_prob = p_on_eff / (p_on_eff + p_off)
        m = p_prod_mRNA * burst_prob / p_deg_mRNA
        protein = max(m * p_prod_prot / p_deg_prot, 0.1)
        protein_levels_sim_data[i] = protein

    return protein_levels_sim_data

def check_system_in_steady_state(simulation_df, gene_params, interaction_matrix, gene_list,
                                  relative_diff_threshold=0.05, relative_slope_threshold=0.01):
    """
    Determines if each gene in the system has reached steady state based on empirical vs theoretical protein levels.

    Args:
        simulation_df (pd.DataFrame): Simulation output with columns like 'time_step' and 'gene_{i}_protein'.
        gene_params (dict): Parameter dictionary for gene kinetics.
        interaction_matrix (np.ndarray): Regulatory matrix (n_genes x n_genes).
        gene_list (list): List of gene names, e.g., ['gene_1', 'gene_2'].
        relative_diff_threshold (float): Threshold for max allowable relative error between empirical and theoretical protein level.
        relative_slope_threshold (float): Threshold for max allowable slope of protein level over time.

    Returns:
        is_steady (bool): True if all genes are in steady state.
        summary_df (pd.DataFrame): Per-gene summary of steady state check.
    """

    n_genes = len(gene_list)
    t_list = sorted(simulation_df['time_step'].unique())
    mean_val = [[] for _ in range(n_genes)]
    gene_means = [[] for _ in range(n_genes)]

    for t in t_list:
        sim_data_t = simulation_df[simulation_df['time_step'] == t]
        steady_state_with_sim_data = steady_state_calc(gene_params, interaction_matrix, gene_list, sim_data=sim_data_t)

        for i in range(n_genes):
            gene_means[i].append(steady_state_with_sim_data[i])
            mean_val[i].append(sim_data_t[f'gene_{i + 1}_protein'].mean())

    t_array = np.array(t_list)
    relative_diffs = []
    relative_slopes = []
    steady_state_flags = []

    for i in range(n_genes):
        empirical = np.array(mean_val[i])
        theoretical = np.array(gene_means[i])

        with np.errstate(divide='ignore', invalid='ignore'):
            relative_diff = np.abs(empirical - theoretical) / theoretical
            relative_diff = np.nan_to_num(relative_diff)
        relative_diffs.append(relative_diff)

        slope, _, _, _, _ = linregress(t_array, empirical)
        final_mean = np.mean(empirical)
        relative_slope = np.abs(slope / final_mean) if final_mean != 0 else 0
        relative_slopes.append(relative_slope)

        is_steady = np.all(relative_diff < relative_diff_threshold) and relative_slope < relative_slope_threshold
        steady_state_flags.append(is_steady)

    summary_df = pd.DataFrame({
        "Gene": [f"Gene {i + 1}" for i in range(n_genes)],
        "Relative Slope": relative_slopes,
        "Steady State?": steady_state_flags
    })

    return all(steady_state_flags), summary_df

def calculate_pairwise_gene_gene_correlation_matrix(simulation_at_t1, gene_list):
    correlations = {}
    for gene_1 in gene_list:
        for gene_2 in gene_list:
            gene_gene_corr = spearmanr(simulation_at_t1[f"{gene_1}_mRNA"], simulation_at_t1[f"{gene_2}_mRNA"]).correlation
            correlations[f"{gene_1}-{gene_2}"] = gene_gene_corr
    correlation_matrix = dict_to_matrix(correlations, gene_list)
    return correlation_matrix

def check_gene_gene_correlation_threshold(pairwise_gene_gene_correlation_matrix, 
                                          gene_list, 
                                          threshold=0.04):
    """
    Splits gene-gene pairs based on absolute correlation threshold.

    Parameters
    ----------
    pairwise_gene_gene_correlation_matrix : pd.DataFrame
        A square matrix of gene-gene correlations (gene × gene).
    gene_list : list of str
        List of gene names, must match the matrix row/column order.
    threshold : float, optional
        Correlation magnitude threshold to separate regulated vs unregulated.
    verbose : bool, optional
        If True, prints gene pairs above the threshold.
    Returns
    -------
    no_regulation : list of tuple
        Gene pairs (i, j) where abs(correlation) ≤ threshold.

    potential_regulation : list of tuple
        Gene pairs (i, j) where abs(correlation) > threshold.
    """
    no_regulation = []
    potential_regulation = []

    for i in range(len(gene_list)):
        for j in range(len(gene_list)):
            if i == j:
                continue  # Skip diagonal
            corr_val = pairwise_gene_gene_correlation_matrix.values[i, j]
            pair = (gene_list[i], gene_list[j])
            if abs(corr_val) > threshold:
                potential_regulation.append(pair)
            else:
                no_regulation.append(pair)

    return no_regulation, potential_regulation

def calculate_pair_correlation(rep_0, rep_1, gene_list, type_comparison="twin"):
    """
    Computes gene-wise pairwise Spearman correlations between delta values across two replicates.

    Parameters
    ----------
    rep_0 : pd.DataFrame
        DataFrame for replicate 1, must include 'clone_id' and '{gene}_mRNA' columns.

    rep_1 : pd.DataFrame
        DataFrame for replicate 2, same structure as rep_0.

    gene_list : list of str
        List of gene names (without "_mRNA" suffix) to analyze.

    type_comparison : str, optional
        Type of comparison:
        - "twin": requires exact matching of `clone_id` between replicates.
        - "random": does not require matching clone_ids.

    Returns
    -------
    correlations : dict
        Dictionary of Spearman correlation values keyed as "gene1-gene2".
        Each value corresponds to correlation of Δgene1 vs Δgene2.
    """
    rep_0 = rep_0.sort_values("clone_id").reset_index(drop=True)
    rep_1 = rep_1.sort_values("clone_id").reset_index(drop=True)

    if type_comparison == "twin":
        if not rep_0["clone_id"].equals(rep_1["clone_id"]):
            raise ValueError("After sorting, clone_ids in rep_0 and rep_1 do not match.")

    correlations = {}
    for gene_1 in gene_list:
        for gene_2 in gene_list:
            delta_1 = rep_0[f"{gene_1}_mRNA"] - rep_1[f"{gene_1}_mRNA"]
            delta_2 = rep_0[f"{gene_2}_mRNA"] - rep_1[f"{gene_2}_mRNA"]
            corr = spearmanr(delta_1, delta_2).correlation
            correlations[f"{gene_1}-{gene_2}"] = corr
    return correlations

def calculate_twin_random_pair_correlations(simulation_single_time, gene_list):
    """
    Computes twin and random pairwise correlation matrices at a given time point.

    Parameters
    ----------
    simulation_single_time : pd.DataFrame
        Subset of simulation data at a single time point. Must contain:
        - 'replicate' column (1 for twin A, 2 for twin B)
        - 'clone_id'
        - '{gene}_mRNA' for each gene in gene_list

    gene_list : list of str
        List of gene names (without "_mRNA" suffix) to analyze.

    Returns
    -------
    twin_correlation_matrix : pd.DataFrame
        Matrix of gene-gene Spearman correlations computed between true twin pairs.

    random_correlation_matrix : pd.DataFrame
        Matrix of gene-gene Spearman correlations between randomly paired clones.
    """
    # Separate replicate 1 and replicate 2
    rep_0 = simulation_single_time[simulation_single_time['replicate'] == 1]
    rep_1 = simulation_single_time[simulation_single_time['replicate'] == 2]

    # Calculate correlations using matched twin pairs
    twin_correlation_dict = calculate_pair_correlation(rep_0, rep_1, gene_list, type_comparison="twin")
    twin_correlation_matrix = dict_to_matrix(twin_correlation_dict, gene_list)

    # Calculate correlations using randomly shuffled pairs
    rep_1_shuffled = rep_1.sample(frac=1, random_state=10100).reset_index(drop=True)
    random_correlation_dict = calculate_pair_correlation(rep_0, rep_1_shuffled, gene_list, type_comparison="random")
    random_correlation_matrix = dict_to_matrix(random_correlation_dict, gene_list)

    return twin_correlation_matrix, random_correlation_matrix

def differentiate_single_state_reg_and_multiple_states(potential_regulation, twin_correlation_matrix, random_correlation_matrix, gene_list, threshold_ratio=25):
    """
    Separates potential regulatory gene pairs into multiple-state vs single-state regulation.

    Parameters
    ----------
    potential_regulation : list of tuple
        List of gene pairs (gene_i, gene_j) with potential regulation.
    twin_correlation_matrix : pd.DataFrame
        Twin pair correlation matrix at time t2.
    random_correlation_matrix : pd.DataFrame
        Random pair correlation matrix at time t2.
    gene_list : list of str
        List of gene names (e.g., 'gene_1') in correct matrix order.
    threshold_ratio : float, optional
        Threshold for abs(random / twin) above which a pair is considered multi-state.
    Returns
    -------
    multiple_states_gene_pairs : list of tuple
        Gene pairs with abs(random / twin) >= threshold_ratio.

    single_state_regulation : list of tuple
        Gene pairs with abs(random / twin) < threshold_ratio.
    """
    multiple_states_gene_pairs = []
    single_state_regulation = []

    for gene_i, gene_j in potential_regulation:
        try:
            t_corr = twin_correlation_matrix.loc[gene_i, gene_j]
            r_corr = random_correlation_matrix.loc[gene_i, gene_j]

            if abs(r_corr / t_corr) >= threshold_ratio:
                multiple_states_gene_pairs.append((gene_i, gene_j))
            else:
                single_state_regulation.append((gene_i, gene_j))
        except ZeroDivisionError:
            # Handle case where twin correlation is 0
            raise ValueError(f"Division by zero for {gene_i} and {gene_j}")
        except KeyError:
            raise ValueError(f"Missing gene pair ({gene_i}, {gene_j}) in correlation matrices.")

    return multiple_states_gene_pairs, single_state_regulation

def identify_reg_if_multiple_states(twin_correlation_matrix_t1, twin_correlation_matrix_t2, random_correlation_matrix_t1, random_correlation_matrix_t2, multiple_states_gene_pairs, gene_list, threshold_relative_increase=0.1):
    """
    Among multiple-state gene pairs, identify which also show regulation
    (based on increased twin correlation from t1 to t2).

    Parameters
    ----------
    twin_correlation_matrix_t1 : pd.DataFrame
        Twin correlation matrix at earlier time t1.

    twin_correlation_matrix_t2 : pd.DataFrame
        Twin correlation matrix at later time t2.

    random_correlation_matrix_t1 : pd.DataFrame
        Random pair correlation matrix at t1 (unused in logic here, included for completeness).

    random_correlation_matrix_t2 : pd.DataFrame
        Random pair correlation matrix at t2 (unused in logic here, included for completeness).

    multiple_states_gene_pairs : list of tuple
        Gene pairs previously classified as showing multiple-state behavior.

    gene_list : list of str
        List of gene names (e.g., 'gene_1').

    threshold_relative_increase : float, optional
        Minimum relative increase in twin correlation from t1 to t2 to call it regulation.

    Returns
    -------
    multiple_states_no_reg : list of tuple
        Gene pairs with multiple states but no significant increase in correlation (no regulation).

    multiple_states_and_reg : list of tuple
        Gene pairs with multiple states and increased correlation (suggesting regulation).
    """
    multiple_states_no_reg = []
    multiple_states_and_reg = []

    for gene_i, gene_j in multiple_states_gene_pairs:
        try:
            corr_t1 = twin_correlation_matrix_t1.loc[gene_i, gene_j]
            corr_t2 = twin_correlation_matrix_t2.loc[gene_i, gene_j]

            if corr_t1 == 0:
                relative_change = np.inf if corr_t2 != 0 else 0
            else:
                relative_change = (corr_t2 - corr_t1) / abs(corr_t1)

            if relative_change > threshold_relative_increase:
                multiple_states_and_reg.append((gene_i, gene_j))
            else:
                multiple_states_no_reg.append((gene_i, gene_j))
        except KeyError:
            raise ValueError(f"Missing gene pair ({gene_i}, {gene_j}) in correlation matrices.")

    return multiple_states_no_reg, multiple_states_and_reg

def get_directions_from_simulation(simulation_t1,
                                   simulation_t2,
                                   gene_pairs,
                                   type_comparison="twin",
                                   threshold=None,
                                   return_raw_and_normalized=True):
    """
    Computes directional Spearman correlations between gene_1 (at t1) and gene_2 (at t2),
    and returns both raw and normalized directional matrices.

    Parameters
    ----------
    simulation_t1 : pd.DataFrame
        Simulation data at time t1, with columns: 'replicate', 'clone_id', '{gene}_mRNA'.

    simulation_t2 : pd.DataFrame
        Simulation data at time t2, same structure as t1.

    gene_pairs : list of tuple
        List of (gene_1, gene_2) pairs to analyze directionally.

    type_comparison : str, optional
        If "twin", checks that clone_ids are aligned. If "random", no check is performed.

    threshold : float or None, optional
        If set, raw correlations with absolute value below this threshold are set to 0.

    return_raw_and_normalized : bool, optional
        If True, returns both raw and normalized correlation matrices.
        Otherwise, returns only normalized.

    Returns
    -------
    raw_matrix : pd.DataFrame
        Raw correlation matrix (gene_1 at t1 → gene_2 at t2).

    normalized_matrix : pd.DataFrame
        Normalized correlation matrix, with 0s where raw matrix was thresholded.
    """
    gene_list = sorted(set(g for pair in gene_pairs for g in pair))

    # Separate replicates for t1 and t2
    rep_0_t1 = simulation_t1[simulation_t1["replicate"] == 1].sort_values("clone_id").reset_index(drop=True)
    rep_1_t1 = simulation_t1[simulation_t1["replicate"] == 2].sort_values("clone_id").reset_index(drop=True)
    rep_0_t2 = simulation_t2[simulation_t2["replicate"] == 1].sort_values("clone_id").reset_index(drop=True)
    rep_1_t2 = simulation_t2[simulation_t2["replicate"] == 2].sort_values("clone_id").reset_index(drop=True)

    all_genes = list(set(gene_1 for gene_1, _ in gene_pairs) | set(gene_2 for _, gene_2 in gene_pairs))
    self_pairs = [(gene, gene) for gene in all_genes if (gene, gene) not in gene_pairs]
    gene_pairs += self_pairs

    if type_comparison == "twin":
        for a, b, label in [
            (rep_0_t1, rep_1_t1, "t1"),
            (rep_0_t2, rep_1_t2, "t2"),
            (rep_0_t1, rep_0_t2, "rep_0 across time"),
            (rep_1_t1, rep_1_t2, "rep_1 across time")
        ]:
            if not a["clone_id"].equals(b["clone_id"]):
                raise ValueError(f"Mismatch in clone_id for {label}")

    # Compute raw directional correlations
    raw_matrix = pd.DataFrame(index=gene_list, columns=gene_list, dtype=float)
    for gene_1, gene_2 in gene_pairs:
        x = rep_0_t1[f"{gene_1}_mRNA"]
        y = rep_1_t2[f"{gene_2}_mRNA"]
        corr = spearmanr(x, y).correlation
        raw_matrix.loc[gene_1, gene_2] = corr
    pre_threshold_raw_matrix = raw_matrix

    # Apply threshold to raw matrix
    if threshold is not None:
        for g1 in gene_list:
            for g2 in gene_list:
                val = raw_matrix.loc[g1, g2]
                if pd.isna(val) or abs(val) < threshold:
                    raw_matrix.loc[g1, g2] = 0.0

    # Compute normalized matrix
    normalized_matrix = pd.DataFrame(index=gene_list, columns=gene_list, dtype=float)
    for g1 in gene_list:
        self_corr = raw_matrix.loc[g1, g1]
        for g2 in gene_list:
            raw_val = raw_matrix.loc[g1, g2]
            if self_corr == 0 or pd.isna(self_corr):
                norm_val = 0.0
            else:
                norm_val = raw_val / self_corr
            if raw_val == 0.0:  # If raw was zeroed by thresholding
                norm_val = 0.0
            normalized_matrix.loc[g1, g2] = norm_val

    if return_raw_and_normalized:
        return raw_matrix, normalized_matrix, pre_threshold_raw_matrix
    else:
        return normalized_matrix


