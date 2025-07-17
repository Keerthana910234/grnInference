#%% Imports
import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.stats import spearmanr
from joblib import Parallel, delayed
import warnings
import gc
import argparse

#%% Config
warnings.filterwarnings('ignore')
N_JOBS = 24
BATCH_SIZE = 500
SAVE_INTERVAL = 500

#%% Utility

def find_csv_files_fast(folder_path):
    """
    Finds all the simulation (*.csv) files in a folder and returns the list.

    Args:
        folder_path (str): Path to the folder to search.

    Returns:
        List[str]: Filenames of all CSV files in the folder (not full paths).
    """
    return [f.name for f in Path(folder_path).glob("df*.csv")]

def extract_param_index(filename):
    """
    Extracts the parameter row index (e.g., '0_1') from a simulation filename.

    This identifies the portion of the filename immediately following 'df_row_',
    stopping before the date stamp (assumed to be in ddmmyyyy format).

    Args:
        filename (str): The simulation filename.

    Returns:
        str: The row identifier (e.g., '0_1'), or 'unknown' if the pattern is not found.
    """
    try:
        core = filename.split("df_row_")[1]
        parts = core.split("_")
        for part in parts:
            if part.isdigit() and len(part) == 8:  # Check for ddmmyyyy format
                return "_".join(parts[:parts.index(part)])
        return "unknown"
    except Exception:
        return "unknown"
#%% Per-file correlation function

def compute_diff_correlation_vectorized(rep1_tf, rep1_target, rep2_tf, rep2_target):
    """
    Computes spearman correlation between the difference of TF and Target between the two replicates.

    Arguments:
        rep1_tf (pd.series): Values of TF in replicate 1
        rep1_target (pd.series): Values of Target in replicate 1
        rep2_tf (pd.series): Values of TF in replicate 2
        rep2_target (pd.series): Values of Target in replicate 2

    Returns:
        float: Spearman correlation between -1 and 1 or np.nan if there were less than 3 points to calculate.

    """
    try:
        diff_tf = rep1_tf - rep2_tf
        diff_target = rep1_target - rep2_target
        if len(diff_tf) < 3:
            return np.nan
        return spearmanr(diff_tf, diff_target)[0]
    except:
        return np.nan

#%%
def get_correlations_optimized(df, time, genes):
    """
    Calculates gene-gene Spearman correlations and replicate-based differential correlations
    (twin vs. random) at a specific time point.

    For each ordered pair of genes (i ≠ j), the function computes:
    - Spearman correlation between gene_i and gene_j across all cells at the given time.
    - Twin differential correlation: correlation between (gene_i_rep1 - gene_i_rep2)
      and (gene_j_rep1 - gene_j_rep2), where replicate 1 and 2 are aligned.
    - Random differential correlation: same as twin, but replicate 2 is randomly shuffled.

    Args:
        df (pd.DataFrame): DataFrame containing columns for 'time_step', 'replicate',
            and expression values for each gene in `genes`.
        time (int): Time point at which to calculate correlations.
        genes (list[str]): List of gene names to compute pairwise correlations.

    Returns:
        list[dict]: Each dict contains:
            - 'gene_1': str, name of the first gene in the pair.
            - 'gene_2': str, name of the second gene in the pair.
            - 'time': int, the time point used.
            - 'gene_gene': float, Spearman correlation between gene_1 and gene_2.
            - 'twin': float, Spearman correlation of replicate-aligned difference.
            - 'random': float, Spearman correlation of shuffled replicate difference.
    """

    df_time = df[df['time_step'] == time]
    replicates = df_time['replicate'].values
    results = []
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            if i == j:
                continue

            gene_i = df_time[genes[i]].values
            gene_j = df_time[genes[j]].values

            corr_gene_gene = spearmanr(gene_i, gene_j)[0] if len(gene_i) >= 3 else np.nan

            rep1_mask = replicates == 1
            rep2_mask = replicates == 2
            if not (rep1_mask.any() and rep2_mask.any()):
                results.append({
                    'gene_1': genes[i], 'gene_2': genes[j], 'time': time,
                    'gene_gene': corr_gene_gene, 'twin': np.nan, 'random': np.nan
                })
                continue

            gi_r1, gj_r1 = gene_i[rep1_mask], gene_j[rep1_mask]
            gi_r2, gj_r2 = gene_i[rep2_mask], gene_j[rep2_mask]
            min_len = min(len(gi_r1), len(gi_r2))
            if min_len < 3:
                results.append({
                    'gene_1': genes[i], 'gene_2': genes[j], 'time': time,
                    'gene_gene': corr_gene_gene, 'twin': np.nan, 'random': np.nan
                })
                continue

            gi_r1, gj_r1 = gi_r1[:min_len], gj_r1[:min_len]
            gi_r2, gj_r2 = gi_r2[:min_len], gj_r2[:min_len]
            twin_corr = compute_diff_correlation_vectorized(gi_r1, gj_r1, gi_r2, gj_r2)
            rand_idx = np.random.permutation(min_len)
            rand_corr = compute_diff_correlation_vectorized(gi_r1, gj_r1, gi_r2[rand_idx], gj_r2[rand_idx])

            results.append({
                'gene_1': genes[i], 'gene_2': genes[j], 'time': time,
                'gene_gene': corr_gene_gene, 'twin': twin_corr, 'random': rand_corr
            })

    return results

#%%
def get_cross_correlations(df, time_points, genes):
    """
    Computes cross-timepoint Spearman correlations between gene pairs across replicates.

    For each unique pair of time points (t1 < t2), the function computes:
    - 'twin_cross': correlation between gene_1 at t1 (replicate 1) and gene_2 at t2 (replicate 2)
    - 'random_cross': same as above, but replicate 2 is randomly shuffled

    Only pairs with at least 3 data points are included.

    Args:
        df (pd.DataFrame): DataFrame containing 'time_step', 'replicate', and gene expression columns.
        time_points (list[int]): List of time points to compute correlations across.
        genes (list[str]): List of gene names to include in pairwise correlations.

    Returns:
        list[dict]: A list of dictionaries with keys:
            - 'gene_1': str, name of gene at t1 (replicate 1)
            - 'gene_2': str, name of gene at t2 (replicate 2)
            - 't1': int, first time point
            - 't2': int, second time point
            - 'twin_cross': float, Spearman correlation across replicates
            - 'random_cross': float, Spearman correlation with shuffled replicate 2
    """

    results = []
    time_pairs = [(time_points[i], time_points[j]) for i in range(len(time_points)) for j in range(i+1, len(time_points))]

    for (t1, t2) in time_pairs:
        df1 = df[df['time_step'] == t1]
        df2 = df[df['time_step'] == t2]
        if df1.empty or df2.empty:
            continue

        rep1_mask = df1['replicate'] == 1
        rep2_mask = df2['replicate'] == 2
        if not (rep1_mask.any() and rep2_mask.any()):
            continue

        for g1 in genes:
            for g2 in genes:
                x = df1[g1][rep1_mask].values
                y = df2[g2][rep2_mask].values
                min_len = min(len(x), len(y))
                if min_len < 3:
                    continue
                x, y = x[:min_len], y[:min_len]
                twin_corr = spearmanr(x, y)[0]
                rand_corr = spearmanr(x, y[np.random.permutation(min_len)])[0]

                results.append({
                    'gene_1': g1, 'gene_2': g2, 't1': t1, 't2': t2,
                    'twin_cross': twin_corr, 'random_cross': rand_corr
                })

    return results

#%%
def process_simulation_stable(sim_info):
    """
    Loads a simulation file and computes timepoint-wise and cross-timepoint correlation metrics.

    This function reads a simulation CSV, extracts gene expression data at specified
    time points, and computes:
    - Gene-gene and replicate-based differential correlations at each time point.
    - Cross-timepoint correlations between gene expression across the specified time points.

    The output includes both correlation sets, each annotated with the simulation's parameter index.

    Args:
        sim_info (tuple): A tuple containing:
            - sim (str): Filename of the simulation CSV.
            - folder (str): Path to the folder containing the file.
            - time_points (list[int]): List of time steps to compute correlations at.
            - genes (list[str]): List of gene names (column names in the CSV).

    Returns:
        dict[str, list[dict]]: A dictionary with two keys:
            - 'correlation': List of dictionaries from `get_correlations_optimized`.
            - 'cross': List of dictionaries from `get_cross_correlations`.

        Returns None if the file does not exist or if processing fails.
    """
    sim, folder, time_points, genes = sim_info
    sim_path = os.path.join(folder, sim)

    if not os.path.exists(sim_path):
        print(f"[Warning] File not found: {sim_path}")
        return None

    try:
        
        df = pd.read_csv(sim_path)

        param_index = extract_param_index(sim)
        corr_all = []
        for t in time_points:
            corr_all.extend(get_correlations_optimized(df, t, genes))
        for r in corr_all:
            r['param_index'] = param_index

        cross_corr = get_cross_correlations(df, time_points, genes)
        for r in cross_corr:
            r['param_index'] = param_index

        del df
        gc.collect()

        return {'correlation': corr_all, 'cross': cross_corr}

    except Exception as e:
        print(f"[Error] Failed to process {sim_path}: {e}")
        print("[Debug] Attempting to read full file to inspect columns...")
        
        try:
            df_debug = pd.read_csv(sim_path)
            print("[Debug] Columns in file:", df_debug.columns.tolist())
        except Exception as inner_e:
            print(f"[Error] Even full read failed for {sim_path}: {inner_e}")
        
        return None

#%%
def main(path_to_simulations, output_folder, genes, time_points):
    files = find_csv_files_fast(path_to_simulations)
    print(f"Found {len(files)} files.")

    all_corr, all_cross = [], []
    c_id, x_id = 0, 0

    os.makedirs(output_folder, exist_ok=True)
    end = len(files)

    for i in range(0,end, BATCH_SIZE):
        batch = files[i:min(i + BATCH_SIZE, end)]
        print(len(batch))
        results = Parallel(n_jobs=N_JOBS)(
            delayed(process_simulation_stable)((f, path_to_simulations, time_points, genes)) for f in batch
        )

        for r in results:
            if not r:
                continue
            all_corr.extend(r['correlation'])
            all_cross.extend(r['cross'])

        if len(all_corr) >= SAVE_INTERVAL:
            pd.DataFrame(all_corr).to_csv(f"{output_folder}/correlation_chunk_{c_id:03d}.csv", index=False)
            print(f"Saved correlation chunk {c_id} with {len(all_corr)} rows")
            all_corr.clear()
            c_id += 1

        if len(all_cross) >= SAVE_INTERVAL:
            pd.DataFrame(all_cross).to_csv(f"{output_folder}/cross_chunk_{x_id:03d}.csv", index=False)
            print(f"Saved cross chunk {x_id} with {len(all_cross)} rows")
            all_cross.clear()
            x_id += 1

    # Final save
    if all_corr:
        pd.DataFrame(all_corr).to_csv(f"{output_folder}/correlation_chunk_{c_id:03d}.csv", index=False)
    if all_cross:
        pd.DataFrame(all_cross).to_csv(f"{output_folder}/cross_chunk_{x_id:03d}.csv", index=False)

# %%
# Example call
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run correlation analysis on Gillespie simulations.")

    parser.add_argument("--path_to_simulations", type=str, required=True,
                        help="Path to folder containing simulation CSVs.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to folder to save analysis results.")
    parser.add_argument("--genes", nargs='+', required=True,
                        help="List of genes to analyze (e.g., gene_1_mRNA gene_2_mRNA).")
    parser.add_argument("--timepoints", nargs='+', type=int, required=True,
                        help="List of time points to measure (e.g., 5 10 15 20).")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok = True)
    main(
        path_to_simulations=args.path_to_simulations,
        output_folder=args.output,
        genes=args.genes,
        time_points=args.timepoints
    )

