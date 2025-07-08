#Code to find correlations for all 3 cases: regulation, no-regulation, mixed-state with regulation and without regulation

#%%
# import pandas as pd
# import numpy as np
# import os
# from scipy.stats import spearmanr
# from tqdm.notebook import tqdm
# from joblib import Parallel, delayed

# #%%
# def find_csv_files(folder_path):
#     csv_files = []
#     for item in os.listdir(folder_path):
#         if item.endswith(".csv"):
#             csv_files.append(item)
#     return csv_files

# # Helper function for correlation
# def compute_diff_correlation(rep1, rep2):
#     diff_tf = rep1['total_TF_mRNA'].values - rep2['total_TF_mRNA'].values
#     diff_target = rep1['total_Target_mRNA'].values - rep2['total_Target_mRNA'].values
#     return spearmanr(diff_tf, diff_target)[0]

# # Compute correlations at a time point
# def get_correlations(df, time):
#     df = df[df['sampling_time'] == time].reset_index(drop=True)
#     corr_gene_gene = spearmanr(df['total_TF_mRNA'], df['total_Target_mRNA'])[0]

#     rep1 = df[df['replicate'] == 0].reset_index(drop=True)
#     rep2 = df[df['replicate'] == 1].reset_index(drop=True)

#     corr_twin_pair = compute_diff_correlation(rep1, rep2)
#     rep2_random = rep2.sample(frac=1, random_state=0, ignore_index=True)
#     corr_random_pair = compute_diff_correlation(rep1, rep2_random)

#     return corr_gene_gene, corr_twin_pair, corr_random_pair

# # Parallelized task per simulation
# def process_simulation_single_population(sim, path_to_folder, t1, t2):
#     sim_path = os.path.join(path_to_folder, sim)
#     population = pd.read_csv(sim_path)

#     param_index = sim.split('_regulation_')[1].split('.csv')[0]

#     c1, c2, c3 = get_correlations(population, t1)
#     c4, c5, c6 = get_correlations(population, t2)

#     return {
#         'parameter_index': param_index,
#         't1_gene_gene_correlation': c1,
#         't1_twin_pair_correlation': c2,
#         't1_random_pair_correlation': c3,
#         't2_gene_gene_correlation': c4,
#         't2_twin_pair_correlation': c5,
#         't2_random_pair_correlation': c6,
#     }

# # Parallelized task per simulation
# def process_simulation_mixed_population(sim, constant_pop, constant_pop_index, path_to_folder, t1, t2):
#     sim_path = os.path.join(path_to_folder, sim)
#     current_pop = pd.read_csv(sim_path)
#     population = pd.concat([current_pop, constant_pop], axis=0)

#     param_index = sim.split('_regulation_')[1].split('.csv')[0]

#     c1, c2, c3 = get_correlations(population, t1)
#     c4, c5, c6 = get_correlations(population, t2)

#     return {
#         'constant_population_index': constant_pop_index,
#         'parameter_index': param_index,
#         't1_gene_gene_correlation': c1,
#         't1_twin_pair_correlation': c2,
#         't1_random_pair_correlation': c3,
#         't2_gene_gene_correlation': c4,
#         't2_twin_pair_correlation': c5,
#         't2_random_pair_correlation': c6,
#     }



# #%%
# param_df = pd.read_csv('/home/mzo5929/Keerthana/grnInference/simulationData/parameters_25000.csv', index_col=0)

# #%%
# path_folder_regulation = "/home/mzo5929/Keerthana/grnInference/simulationData/large_scale_parameter_scan/regulation/"
# list_of_files_regulation = find_csv_files(path_folder_regulation)

# correlation_list = []
# t1 = 300
# t2 = 600

# # Run in parallel with progress bar
# correlation_list = Parallel(n_jobs=32)(
#     delayed(process_simulation_single_population)(sim, path_folder_regulation, t1, t2) for sim in tqdm(list_of_files_regulation)
# )

# # Build final DataFrame
# correlation_df_k_on_tf = pd.DataFrame(correlation_list)


#%%
# import pandas as pd
# import numpy as np
# import os
# from pathlib import Path
# from scipy.stats import spearmanr
# from joblib import Parallel, delayed
# from tqdm.auto import tqdm
# import time
# import multiprocessing as mp
# import warnings
# import gc
# warnings.filterwarnings('ignore')

# # Conservative but stable configuration
# N_JOBS = 32  # Even more conservative to prevent stalls
# BATCH_SIZE = 500  # Smaller chunks for more frequent updates
# print(f"Using {N_JOBS} workers for processing")

# def find_csv_files_fast(folder_path):
#     """Faster file discovery using pathlib"""
#     return [f.name for f in Path(folder_path).glob("*.csv")]

# def compute_diff_correlation_vectorized(rep1_tf, rep1_target, rep2_tf, rep2_target):
#     """Vectorized correlation computation"""
#     try:
#         diff_tf = rep1_tf - rep2_tf
#         diff_target = rep1_target - rep2_target
#         if len(diff_tf) < 3:  # Need at least 3 points for correlation
#             return np.nan
#         return spearmanr(diff_tf, diff_target)[0]
#     except:
#         return np.nan

# def get_correlations_optimized(df, time):
#     """Optimized correlation computation with error handling"""
#     try:
#         # Filter once and work with views/references
#         time_mask = df['sampling_time'] == time
#         df_time = df[time_mask]
        
#         if df_time.empty or len(df_time) < 3:
#             return np.nan, np.nan, np.nan
        
#         # Extract arrays once to avoid repeated column access
#         tf_values = df_time['total_TF_mRNA'].values
#         target_values = df_time['total_Target_mRNA'].values
#         replicates = df_time['replicate'].values
        
#         # Gene-gene correlation
#         if len(tf_values) < 3:
#             corr_gene_gene = np.nan
#         else:
#             corr_gene_gene = spearmanr(tf_values, target_values)[0]
        
#         # Split by replicate using boolean indexing
#         rep0_mask = replicates == 0
#         rep1_mask = replicates == 1
        
#         if not (rep0_mask.any() and rep1_mask.any()):
#             return corr_gene_gene, np.nan, np.nan
        
#         rep1_tf = tf_values[rep0_mask]
#         rep1_target = target_values[rep0_mask]
#         rep2_tf = tf_values[rep1_mask]
#         rep2_target = target_values[rep1_mask]
        
#         # Ensure same length for correlation
#         min_len = min(len(rep1_tf), len(rep2_tf))
#         if min_len < 3:
#             return corr_gene_gene, np.nan, np.nan
        
#         rep1_tf = rep1_tf[:min_len]
#         rep1_target = rep1_target[:min_len]
#         rep2_tf = rep2_tf[:min_len]
#         rep2_target = rep2_target[:min_len]
        
#         # Twin pair correlation
#         corr_twin_pair = compute_diff_correlation_vectorized(
#             rep1_tf, rep1_target, rep2_tf, rep2_target
#         )
        
#         # Random pair correlation with fixed seed for reproducibility
#         np.random.seed(0)
#         random_indices = np.random.permutation(len(rep2_tf))
#         rep2_tf_random = rep2_tf[random_indices]
#         rep2_target_random = rep2_target[random_indices]
        
#         corr_random_pair = compute_diff_correlation_vectorized(
#             rep1_tf, rep1_target, rep2_tf_random, rep2_target_random
#         )
        
#         return corr_gene_gene, corr_twin_pair, corr_random_pair
    
#     except Exception as e:
#         return np.nan, np.nan, np.nan

# def process_simulation_stable(sim_info):
#     """Stable simulation processing with extensive error handling"""
#     sim, path_to_folder, t1, t2 = sim_info
    
#     try:
#         sim_path = os.path.join(path_to_folder, sim)
        
#         # Check if file exists
#         if not os.path.exists(sim_path):
#             return None
        
#         # Read CSV with error handling
#         population = pd.read_csv(
#             sim_path,
#             usecols=['sampling_time', 'replicate', 'total_TF_mRNA', 'total_Target_mRNA'],
#             dtype={
#                 'sampling_time': np.int16,
#                 'replicate': np.int8,
#                 'total_TF_mRNA': np.float32,
#                 'total_Target_mRNA': np.float32
#             },
#             engine='c',
#             on_bad_lines='skip'  # Skip problematic lines
#         )
        
#         if population.empty:
#             return None
        
#         # Extract parameter index more efficiently
#         try:
#             param_index = sim.split('_regulation_')[1].replace('.csv', '')
#         except:
#             param_index = sim.replace('.csv', '')
        
#         # Compute correlations with error handling
#         c1, c2, c3 = get_correlations_optimized(population, t1)
#         c4, c5, c6 = get_correlations_optimized(population, t2)
        
#         # Explicit cleanup
#         del population
#         gc.collect()  # Force garbage collection
        
#         return {
#             'parameter_index': param_index,
#             't1_gene_gene_correlation': c1,
#             't1_twin_pair_correlation': c2,
#             't1_random_pair_correlation': c3,
#             't2_gene_gene_correlation': c4,
#             't2_twin_pair_correlation': c5,
#             't2_random_pair_correlation': c6,
#         }
        
#     except Exception as e:
#         # Don't print every error to avoid spam
#         return None

# def process_chunk_stable(chunk_info):
#     """Process files in chunks with progress tracking"""
#     start_idx, end_idx, file_list, path_folder_regulation, t1, t2 = chunk_info
    
#     chunk_files = file_list[start_idx:end_idx]
#     results = []
#     chunk_num = start_idx//BATCH_SIZE + 1
    
#     print(f"Starting chunk {chunk_num} ({len(chunk_files)} files)")
#     start_time = time.time()
    
#     for i, sim in enumerate(chunk_files):
#         result = process_simulation_stable((sim, path_folder_regulation, t1, t2))
#         if result is not None:
#             results.append(result)
        
#         # Print progress every 5 files
#         if (i + 1) % 25 == 0:
#             elapsed = time.time() - start_time
#             rate = (i + 1) / elapsed
#             print(f"  Chunk {chunk_num}: {i+1}/{len(chunk_files)} files, {rate:.1f} files/sec")
    
#     elapsed = time.time() - start_time
#     success_rate = len(results) / len(chunk_files) * 100
#     print(f"Completed chunk {chunk_num}: {len(results)}/{len(chunk_files)} files ({success_rate:.1f}% success) in {elapsed:.1f}s")
    
#     return results

# def main():
#     """Main execution function with chunked processing"""
#     print("Starting correlation analysis...")
    
#     # Configuration
#     path_folder_regulation = "/home/mzo5929/Keerthana/grnInference/simulationData/large_scale_parameter_scan/regulation/"
#     t1, t2 = 300, 600
    
#     # Find files faster
#     print("Finding CSV files...")
#     list_of_files_regulation = find_csv_files_fast(path_folder_regulation)
#     print(f"Found {len(list_of_files_regulation)} files")
    
#     # Create chunks for processing
#     chunks = []
#     for i in range(0, len(list_of_files_regulation), BATCH_SIZE):
#         chunks.append((
#             i, 
#             min(i + BATCH_SIZE, len(list_of_files_regulation)),
#             list_of_files_regulation,
#             path_folder_regulation,
#             t1, 
#             t2
#         ))
    
#     print(f"Processing {len(chunks)} chunks with {N_JOBS} workers...")
#     print("Progress will be shown for each chunk...")
    
#     # Process chunks in parallel
#     all_results = []
#     start_time = time.time()
    
#     try:
#         chunk_results = Parallel(
#             n_jobs=N_JOBS,
#             backend='multiprocessing',
#             verbose=2,  # More verbose for interactive
#             timeout=600  # 10 minute timeout per chunk
#         )(
#             delayed(process_chunk_stable)(chunk_info) 
#             for chunk_info in chunks
#         )
        
#         # Flatten results
#         for chunk_result in chunk_results:
#             if chunk_result:
#                 all_results.extend(chunk_result)
        
#         total_time = time.time() - start_time
                
#     except Exception as e:
#         print(f"Error in parallel processing: {e}")
#         return pd.DataFrame()
    
#     # Build final DataFrame
#     if all_results:
#         correlation_df_k_on_tf = pd.DataFrame(all_results)
#         print(f"Successfully processed {len(correlation_df_k_on_tf)} files")
#     else:
#         correlation_df_k_on_tf = pd.DataFrame()
#         print("No files processed successfully")
    
#     return correlation_df_k_on_tf

# if __name__ == "__main__":
#     # Execute the main function
#     correlation_df_k_on_tf = main()
    
#     if not correlation_df_k_on_tf.empty:
#         # Save results
#         output_file = 'correlation_results_optimized.csv'
#         correlation_df_k_on_tf.to_csv(output_file, index=False)
#         print(f"Results saved to {output_file}")
        
#         # Display summary statistics
#         print("\nSummary:")
#         print(f"Total files processed: {len(correlation_df_k_on_tf)}")
#         print(f"Columns: {list(correlation_df_k_on_tf.columns)}")
#     else:
#         print("No results to save")
    
#%%
import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.stats import spearmanr
from joblib import Parallel, delayed
import time
import gc
import warnings
import random

warnings.filterwarnings('ignore')

# Configuration
N_JOBS = 24
BATCH_SIZE = 500
SAVE_INTERVAL = 500  # Save every 1000 rows


def find_csv_files_fast(folder_path):
    return [f.name for f in Path(folder_path).glob("*.csv")]


def compute_diff_correlation_vectorized(rep1_tf, rep1_target, rep2_tf, rep2_target):
    try:
        diff_tf = rep1_tf - rep2_tf
        diff_target = rep1_target - rep2_target
        if len(diff_tf) < 3:
            return np.nan
        return spearmanr(diff_tf, diff_target)[0]
    except:
        return np.nan


def get_correlations_optimized(df, time):
    try:
        df_time = df[df['sampling_time'] == time]
        if df_time.empty or len(df_time) < 3:
            return np.nan, np.nan, np.nan

        tf_values = df_time['total_TF_mRNA'].values
        target_values = df_time['total_Target_mRNA'].values
        replicates = df_time['replicate'].values

        corr_gene_gene = spearmanr(tf_values, target_values)[0] if len(tf_values) >= 3 else np.nan

        rep0_mask = replicates == 0
        rep1_mask = replicates == 1
        if not (rep0_mask.any() and rep1_mask.any()):
            return corr_gene_gene, np.nan, np.nan

        rep1_tf = tf_values[rep0_mask]
        rep1_target = target_values[rep0_mask]
        rep2_tf = tf_values[rep1_mask]
        rep2_target = target_values[rep1_mask]

        min_len = min(len(rep1_tf), len(rep2_tf))
        if min_len < 3:
            return corr_gene_gene, np.nan, np.nan

        rep1_tf = rep1_tf[:min_len]
        rep1_target = rep1_target[:min_len]
        rep2_tf = rep2_tf[:min_len]
        rep2_target = rep2_target[:min_len]

        corr_twin_pair = compute_diff_correlation_vectorized(rep1_tf, rep1_target, rep2_tf, rep2_target)

        np.random.seed(0)
        random_indices = np.random.permutation(len(rep2_tf))
        rep2_tf_random = rep2_tf[random_indices]
        rep2_target_random = rep2_target[random_indices]

        corr_random_pair = compute_diff_correlation_vectorized(rep1_tf, rep1_target, rep2_tf_random, rep2_target_random)

        return corr_gene_gene, corr_twin_pair, corr_random_pair

    except:
        return np.nan, np.nan, np.nan

def get_cross_correlations(df, t1, t2):
    """
    Calculate cross-correlations between genes at different time points.
    Each replicate is used at only one time point.
    Returns direct spearman correlations (not difference-based).
    """
    try:
        # Get data for both time points
        df_t1 = df[df['sampling_time'] == t1]
        df_t2 = df[df['sampling_time'] == t2]
        
        if df_t1.empty or df_t2.empty or len(df_t1) < 3 or len(df_t2) < 3:
            return {
                'TF_t1_Target_t2_twin': np.nan,
                'TF_t1_Target_t2_random': np.nan,
                'Target_t1_TF_t2_twin': np.nan,
                'Target_t1_TF_t2_random': np.nan
            }

        # Extract values
        tf_t1 = df_t1['total_TF_mRNA'].values
        target_t1 = df_t1['total_Target_mRNA'].values
        replicates_t1 = df_t1['replicate'].values
        
        tf_t2 = df_t2['total_TF_mRNA'].values
        target_t2 = df_t2['total_Target_mRNA'].values
        replicates_t2 = df_t2['replicate'].values

        # Check if we have both replicates at both time points
        rep0_mask_t1 = replicates_t1 == 0
        rep1_mask_t2 = replicates_t2 == 1
        
        if not (rep0_mask_t1.any() and rep1_mask_t2.any()):
            return {
                'TF_t1_Target_t2_twin': np.nan,
                'TF_t1_Target_t2_random': np.nan,
                'Target_t1_TF_t2_twin': np.nan,
                'Target_t1_TF_t2_random': np.nan
            }
        
        # Get values for rep0 at t1 and rep1 at t2
        rep0_tf_t1 = tf_t1[rep0_mask_t1]
        rep0_target_t1 = target_t1[rep0_mask_t1]
        rep1_tf_t2 = tf_t2[rep1_mask_t2]
        rep1_target_t2 = target_t2[rep1_mask_t2]
        
        min_len = min(len(rep0_tf_t1), len(rep1_tf_t2))
        if min_len < 3:
            return {
                'TF_t1_Target_t2_twin': np.nan,
                'TF_t1_Target_t2_random': np.nan,
                'Target_t1_TF_t2_twin': np.nan,
                'Target_t1_TF_t2_random': np.nan
            }
        
        # Truncate to same length
        rep0_tf_t1 = rep0_tf_t1[:min_len]
        rep0_target_t1 = rep0_target_t1[:min_len]
        rep1_tf_t2 = rep1_tf_t2[:min_len]
        rep1_target_t2 = rep1_target_t2[:min_len]
        
        # === TF(t1) -> Target(t2) CORRELATIONS ===
        # Twin pair: Direct correlation between rep0 TF at t1 and rep1 Target at t2
        cross_corr_tf_target_twin = spearmanr(rep0_tf_t1, rep1_target_t2)[0]
        
        # Random pair: Same but with shuffled Target at t2
        np.random.seed(0)
        random_indices_1 = np.random.permutation(len(rep1_target_t2))
        rep1_target_t2_random = rep1_target_t2[random_indices_1]
        cross_corr_tf_target_random = spearmanr(rep0_tf_t1, rep1_target_t2_random)[0]
        
        # === Target(t1) -> TF(t2) CORRELATIONS ===
        # Twin pair: Direct correlation between rep0 Target at t1 and rep1 TF at t2
        cross_corr_target_tf_twin = spearmanr(rep0_target_t1, rep1_tf_t2)[0]
        
        # Random pair: Same but with shuffled TF at t2
        np.random.seed(1)  # Different seed
        random_indices_2 = np.random.permutation(len(rep1_tf_t2))
        rep1_tf_t2_random = rep1_tf_t2[random_indices_2]
        cross_corr_target_tf_random = spearmanr(rep0_target_t1, rep1_tf_t2_random)[0]
        
        return {
            'TF_t1_Target_t2_twin': cross_corr_tf_target_twin,
            'TF_t1_Target_t2_random': cross_corr_tf_target_random,
            'Target_t1_TF_t2_twin': cross_corr_target_tf_twin,
            'Target_t1_TF_t2_random': cross_corr_target_tf_random
        }

    except Exception as e:
        print(f"Error in cross-correlation calculation: {e}")
        return {
            'TF_t1_Target_t2_twin': np.nan,
            'TF_t1_Target_t2_random': np.nan,
            'Target_t1_TF_t2_twin': np.nan,
            'Target_t1_TF_t2_random': np.nan
        }

def process_simulation_stable(sim_info):
    sim, path_to_folder, t1, t2 = sim_info
    try:
        sim_path = os.path.join(path_to_folder, sim)
        if not os.path.exists(sim_path):
            return None

        population = pd.read_csv(
            sim_path,
            usecols=['sampling_time', 'replicate', 'total_TF_mRNA', 'total_Target_mRNA'],
            dtype={
                'sampling_time': np.int16,
                'replicate': np.int8,
                'total_TF_mRNA': np.float32,
                'total_Target_mRNA': np.float32
            },
            engine='c',
            on_bad_lines='skip'
        )
        param_index = sim.split('two_way_regulation_')[1].replace('.csv', '') if 'two_way_regulation_' in sim else 0
        # print(f"Processing simulation: {sim}, Parameter index: {param_index}")
        # print("123!!!")
        c1, c2, c3 = get_correlations_optimized(population, t1)
        c4, c5, c6 = get_correlations_optimized(population, t2)
        required_data = {}
        correlation_data = {
            'param_index': param_index,
            't1_gene_gene_correlation': c1,
            't1_twin_pair_correlation': c2,
            't1_random_pair_correlation': c3,
            't2_gene_gene_correlation': c4,
            't2_twin_pair_correlation': c5,
            't2_random_pair_correlation': c6,
        }
        required_data['correlation_data'] = correlation_data
        cross_correlation = get_cross_correlations(population, t1, t2)
        cross_correlation['param_index'] = param_index
        del population
        required_data['cross_correlation'] = cross_correlation
        gc.collect()

        return required_data
    except Exception as e:
        print(f"Error processing simulation: {e} in {sim}")
        return None


def process_simulation_two_population(sim_info):
    sim1, sim2, path_to_folder, t1, t2 = sim_info
    try:
        sim_path_1 = os.path.join(path_to_folder, sim1)
        sim_path_2 = os.path.join(path_to_folder, sim2)
        if not (os.path.exists(sim_path_1) and os.path.exists(sim_path_2)):
            return None

        population_1 = pd.read_csv(
            sim_path_1,
            usecols=['sampling_time', 'replicate', 'total_TF_mRNA', 'total_Target_mRNA'],
            dtype={
                'sampling_time': np.int16,
                'replicate': np.int8,
                'total_TF_mRNA': np.float32,
                'total_Target_mRNA': np.float32
            },
            engine='c',
            on_bad_lines='skip'
        )
        population_2 = pd.read_csv(
            sim_path_2,
            usecols=['sampling_time', 'replicate', 'total_TF_mRNA', 'total_Target_mRNA'],
            dtype={
                'sampling_time': np.int16,
                'replicate': np.int8,
                'total_TF_mRNA': np.float32,
                'total_Target_mRNA': np.float32
            },
            engine='c',
            on_bad_lines='skip'
        )

        if population_1.empty or population_2.empty:
            return None

        population = pd.concat([population_1, population_2], axis=0, ignore_index=True)
        del population_1, population_2 
        def find_param_index(sim):
            if "without_regulation_" in sim:
                try:
                    param_index = sim.split('without_regulation_')[1].replace('.csv', '')
                except:
                    param_index = sim.replace('.csv', '')
            else:
                try:
                    param_index = sim.split('_regulation_')[1].replace('.csv', '')
                except:
                    param_index = sim.replace('.csv', '')
            return param_index
        param_1 = find_param_index(sim1)
        param_2 = find_param_index(sim2)
        c1, c2, c3 = get_correlations_optimized(population, t1)
        c4, c5, c6 = get_correlations_optimized(population, t2)
        required_data = {}
        correlation_data = {
            'parameter_index_1': param_1,
            'parameter_index_2': param_2,
            't1_gene_gene_correlation': c1,
            't1_twin_pair_correlation': c2,
            't1_random_pair_correlation': c3,
            't2_gene_gene_correlation': c4,
            't2_twin_pair_correlation': c5,
            't2_random_pair_correlation': c6,
        }
        required_data['correlation_data'] = correlation_data
        cross_correlation = get_cross_correlations(population, t1, t2)
        cross_correlation['parameter_index_1'] = param_1
        cross_correlation['parameter_index_2'] = param_2
        del population
        required_data['cross_correlation'] = cross_correlation
        gc.collect()

        return required_data
    except:
        return None

def generate_file_pairs(file_list, n_pairs, processed_pairs=None, seed=42):
    """
    For very large pair requirements, use systematic sampling.
    """
    if processed_pairs is None:
        processed_pairs = set()
    
    random.seed(seed)
    n_files = len(file_list)
    file_pairs = []
    
    print(f"Using systematic approach for {n_pairs} pairs from {n_files} files...")
    
    # Create a more systematic approach
    # Divide files into chunks and sample across chunks
    chunk_size = max(100, n_files // 100)  # Reasonable chunk size
    
    generated_count = 0
    attempt_count = 0
    max_attempts = n_pairs * 3
    
    while generated_count < n_pairs and attempt_count < max_attempts:
        # Pick two random indices
        i, j = random.sample(range(n_files), 2)
        
        file1, file2 = file_list[i], file_list[j]
        pair_id = f"{min(file1, file2)}_{max(file1, file2)}"
        
        if pair_id not in processed_pairs:
            file_pairs.append((file1, file2))
            processed_pairs.add(pair_id)
            generated_count += 1
            
            if generated_count % 10000 == 0:
                print(f"Generated {generated_count} pairs...")
        
        attempt_count += 1
    
    print(f"Systematic generation complete: {len(file_pairs)} pairs")
    return file_pairs

def main_two_populations(path_folder_regulation, output_prefix, n_pairs=25000):
    t1, t2 = 300, 600
    list_of_files_regulation = find_csv_files_fast(path_folder_regulation)
    
    # Generate file pairs (your existing logic)
    list_pairs_of_files = generate_file_pairs(list_of_files_regulation, n_pairs)
    # current_calculated_pairs = pd.read_csv("/home/mzo5929/Keerthana/grnInference/analysisData2/large_scale_parameter_scan/correlation_df_large_scale_parameter_scan_two_population_no_regulation.csv")
    print(f"Generated {len(list_pairs_of_files)} file pairs to process.")
    # for pair in list_pairs_of_files:
    #     index_1 = int(pair[0].split('regulation_')[1].replace('.csv', ''))
    #     index_2 = int(pair[1].split('regulation_')[1].replace('.csv', ''))
    #     if ((index_1, index_2) in zip(current_calculated_pairs['parameter_index_1'], current_calculated_pairs['parameter_index_2'])) or \
    #        ((index_2, index_1) in zip(current_calculated_pairs['parameter_index_1'], current_calculated_pairs['parameter_index_2'])):
    #         list_pairs_of_files.remove(pair)
    print(f"After filtering, {len(list_pairs_of_files)} pairs remain to process.")
    # Separate result lists
    all_correlation_results = []
    all_cross_correlation_results = []
    correlation_chunk_id = 0
    cross_correlation_chunk_id = 0
    
    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix, exist_ok=True)
    
    for i in range(0, len(list_pairs_of_files), BATCH_SIZE):
        print(f"Processing files {i} to {min(i + BATCH_SIZE, len(list_pairs_of_files))}...")
        chunk_files = list_pairs_of_files[i:i + BATCH_SIZE]
        
        chunk_results = Parallel(n_jobs=N_JOBS)(
            delayed(process_simulation_two_population)((sim_1, sim_2, path_folder_regulation, t1, t2))
            for (sim_1, sim_2) in chunk_files
        )
        
        print(f"Processed chunk {i//BATCH_SIZE + 1}: {len(chunk_results)} pairs")
        chunk_results = [res for res in chunk_results if res is not None]
        
        # Separate correlation and cross-correlation data
        correlation_batch = []
        cross_correlation_batch = []
        
        for result in chunk_results:
            if 'correlation_data' in result:
                correlation_batch.append(result['correlation_data'])
            
            if 'cross_correlation' in result:
                cross_correlation_batch.append(result['cross_correlation'])
        
        # Add to main result lists
        all_correlation_results.extend(correlation_batch)
        all_cross_correlation_results.extend(cross_correlation_batch)
        
        # Save correlation results if threshold reached
        if len(all_correlation_results) >= SAVE_INTERVAL:
            correlation_file_path = f'{output_prefix}correlation_partial_{correlation_chunk_id:03d}.csv'
            correlation_df = pd.DataFrame(all_correlation_results)
            correlation_df.to_csv(correlation_file_path, index=False)
            print(f"Saved correlation chunk {correlation_chunk_id}: {len(all_correlation_results)} records")
            correlation_chunk_id += 1
            all_correlation_results.clear()
        
        # Save cross-correlation results if threshold reached
        if len(all_cross_correlation_results) >= SAVE_INTERVAL:
            cross_correlation_file_path = f'{output_prefix}cross_correlation_partial_{cross_correlation_chunk_id:03d}.csv'
            cross_correlation_df = pd.DataFrame(all_cross_correlation_results)
            cross_correlation_df.to_csv(cross_correlation_file_path, index=False)
            print(f"Saved cross-correlation chunk {cross_correlation_chunk_id}: {len(all_cross_correlation_results)} records")
            cross_correlation_chunk_id += 1
            all_cross_correlation_results.clear()
    
    # Save remaining correlation results
    if all_correlation_results:
        correlation_file_path = f'{output_prefix}correlation_partial_{correlation_chunk_id:03d}.csv'
        correlation_df = pd.DataFrame(all_correlation_results)
        correlation_df.to_csv(correlation_file_path, index=False)
        print(f"Saved final correlation chunk {correlation_chunk_id}: {len(all_correlation_results)} records")
    
    # Save remaining cross-correlation results
    if all_cross_correlation_results:
        cross_correlation_file_path = f'{output_prefix}cross_correlation_partial_{cross_correlation_chunk_id:03d}.csv'
        cross_correlation_df = pd.DataFrame(all_cross_correlation_results)
        cross_correlation_df.to_csv(cross_correlation_file_path, index=False)
        print(f"Saved final cross-correlation chunk {cross_correlation_chunk_id}: {len(all_cross_correlation_results)} records")
    return 

def main(path_folder_regulation, output_prefix):
    t1, t2 = 300, 600
    list_of_files_regulation = find_csv_files_fast(path_folder_regulation)
    # current_results = pd.read_csv("/home/mzo5929/Keerthana/grnInference/analysisData2/large_scale_parameter_scan/correlation_df_large_scale_parameter_scan_modified_regulation.csv")
    # update_list_of_files = []
    # for file in list_of_files_regulation:
    #     index = int(file.split('modified_regulation_')[1].replace('.csv', ''))
    #     if index not in current_results['parameter_index'].values:
    #         update_list_of_files.append(file)
    update_list_of_files = list_of_files_regulation.copy()
    print(f"Found {len(update_list_of_files)} files to process out of {len(list_of_files_regulation)} total files.")
    all_correlation_results = []
    all_cross_correlation_results = []
    correlation_chunk_id = 0
    cross_correlation_chunk_id = 0

    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix, exist_ok=True)
    
    for i in range(0, len(update_list_of_files), BATCH_SIZE):
        #Check if this parameter index has already been processed and is in current_results


        print(f"Processing files {i} to {min(i + BATCH_SIZE, len(update_list_of_files))}...")
        chunk_files = update_list_of_files[i:i + BATCH_SIZE]
        chunk_results = Parallel(n_jobs=N_JOBS)(
            delayed(process_simulation_stable)((sim, path_folder_regulation, t1, t2))
            for sim in chunk_files
        )
        chunk_results = [res for res in chunk_results if res is not None]
        correlation_batch = []
        cross_correlation_batch = []
        
        for result in chunk_results:
            if 'correlation_data' in result:
                correlation_batch.append(result['correlation_data'])
            
            if 'cross_correlation' in result:
                cross_correlation_batch.append(result['cross_correlation'])
        
        # Add to main result lists
        all_correlation_results.extend(correlation_batch)
        all_cross_correlation_results.extend(cross_correlation_batch)


        if len(all_correlation_results) >= SAVE_INTERVAL:
            correlation_file_path = f'{output_prefix}correlation_partial_{correlation_chunk_id:03d}.csv'
            correlation_df = pd.DataFrame(all_correlation_results)
            correlation_df.to_csv(correlation_file_path, index=False)
            print(f"Saved correlation chunk {correlation_chunk_id}: {len(all_correlation_results)} records")
            correlation_chunk_id += 1
            all_correlation_results.clear()
        
        # Save cross-correlation results if threshold reached
        if len(all_cross_correlation_results) >= SAVE_INTERVAL:
            cross_correlation_file_path = f'{output_prefix}cross_correlation_partial_{cross_correlation_chunk_id:03d}.csv'
            cross_correlation_df = pd.DataFrame(all_cross_correlation_results)
            cross_correlation_df.to_csv(cross_correlation_file_path, index=False)
            print(f"Saved cross-correlation chunk {cross_correlation_chunk_id}: {len(all_cross_correlation_results)} records")
            cross_correlation_chunk_id += 1
            all_cross_correlation_results.clear()

    # Save remaining
    if all_correlation_results:
        correlation_file_path = f'{output_prefix}correlation_partial_{correlation_chunk_id:03d}.csv'
        correlation_df = pd.DataFrame(all_correlation_results)
        correlation_df.to_csv(correlation_file_path, index=False)
        print(f"Saved final correlation chunk {correlation_chunk_id}: {len(all_correlation_results)} records")
    
    # Save remaining cross-correlation results
    if all_cross_correlation_results:
        cross_correlation_file_path = f'{output_prefix}cross_correlation_partial_{cross_correlation_chunk_id:03d}.csv'
        cross_correlation_df = pd.DataFrame(all_cross_correlation_results)
        cross_correlation_df.to_csv(cross_correlation_file_path, index=False)
        print(f"Saved final cross-correlation chunk {cross_correlation_chunk_id}: {len(all_cross_correlation_results)} records")
    return 

#%%
#Analyze simulation with with 2 state populations and no regulation.


#%%
# Example usage:
main("/home/mzo5929/Keerthana/grnInference/simulationData/large_scale_parameter_scan/two_way_regulation_new/", "/home/mzo5929/Keerthana/grnInference/analysisData2/large_scale_parameter_scan/two_way_regulation_new/")
# main("/home/mzo5929/Keerthana/grnInference/simulationData/large_scale_parameter_scan/without_regulation/", "/home/mzo5929/Keerthana/grnInference/analysisData2/large_scale_parameter_scan/no_regulation/")

# #%%
# main("/home/mzo5929/Keerthana/grnInference/simulationData/large_scale_parameter_scan/modified_regulation/", "/home/mzo5929/Keerthana/grnInference/analysisData2/large_scale_parameter_scan/modified_regulation/")

# #%%
# main_two_populations("/home/mzo5929/Keerthana/grnInference/simulationData/large_scale_parameter_scan/without_regulation/","/home/mzo5929/Keerthana/grnInference/analysisData2/large_scale_parameter_scan/two_population_no_regulation/" )
#%%
# main_two_populations("/home/mzo5929/Keerthana/grnInference/simulationData/large_scale_parameter_scan/regulation/","/home/mzo5929/Keerthana/grnInference/analysisData2/large_scale_parameter_scan/two_population_regulation/" )