import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import random
import sys
import json
from stochastic_bursting_simulator import simulate
from stochastic_bursting_simulator_variations import simulate_asynchronous
from stochastic_bursting_simulator_variations import simulate_both_TF, simulate_self_regulation
from multiprocessing import Pool
import multiprocessing
from time import sleep
import os

#Median values for all parameters
#Median
params_median = {'k_on_TF': 0.27,
          'k_off_TF': 8.4,
          'burst_size_TF': 32,
          'k_on_Target': 0.25,
          'k_off_Target': 7.7,
          'burst_size_Target': 40,
          'splicing_half_life_minutes': 1, #7,
          'mrna_half_life_TF': 2.5,
          'mrna_half_life_Target': 3.7,
          'protein_half_life': 28, #28, 
          'n':2,
          'protein_production_rate': 0.059, 
          'labeling_efficiency': 1,
          'pulse_time': 60,
          'num_cells': 20_000,
          'dynamics': 'MM',
          'capture_efficiency': 1}

#Lower quartile parameter values
params_lower_quartile = {'k_on_TF': 0.47,
          'k_off_TF': 8.4,
          'burst_size_TF': 32,
          'k_on_Target': 0.42,
          'k_off_Target': 7.7,
          'burst_size_Target': 40,
          'splicing_half_life_minutes': 1, #7,
          'mrna_half_life_TF': 2.5,
          'mrna_half_life_Target': 3.7,
          'protein_half_life': 28, #28, 
          'protein_production_rate': 0.059, 
          'labeling_efficiency': 1,
          'pulse_time': 60,
          'num_cells': 20_000,
          'dynamics': 'MM',
          'capture_efficiency': 1}

params_upper_quartile = {'k_on_TF': 0.14,
          'k_off_TF': 8.4,
          'burst_size_TF': 32,
          'k_on_Target': 0.13,
          'k_off_Target': 7.7,
          'burst_size_Target': 40,
          'splicing_half_life_minutes': 1, #7,
          'mrna_half_life_TF': 2.5,
          'mrna_half_life_Target': 3.7,
          'protein_half_life': 28, #28, 
           
          'protein_production_rate': 0.059, 
          'labeling_efficiency': 1,
          'pulse_time': 60,
          'num_cells': 20_000,
          'dynamics': 'MM',
          'capture_efficiency': 1}
params_lower_quartile_both_TF = {
    'k_on_TF': 0.47,
    'k_off_TF': 8.4,
    'burst_size_TF': 32,
    'k_on_Target': 0.42,
    'k_off_Target': 7.7,
    'burst_size_Target': 40,
    'splicing_half_life_minutes': 1,
    'mrna_half_life_TF': 2.5,
    'mrna_half_life_Target': 3.7,
    'TF_protein_half_life': 28,
    'Target_protein_half_life': 28,
    'TF_protein_production_rate': 0.059,
    'Target_protein_production_rate': 0.059,
    'labeling_efficiency': 1,
    'pulse_time': 60,
    'num_cells': 20,
    'dynamics': 'MM',
    'capture_efficiency': 1,
    'n': 1  # Hill coefficient - add this if needed
}

params_median_both_TF = {
    'k_on_TF': 0.27,
    'k_off_TF': 8.4,
    'burst_size_TF': 32,
    'k_on_Target': 0.25,
    'k_off_Target': 7.7,
    'burst_size_Target': 40,
    'splicing_half_life_minutes': 7,
    'mrna_half_life_TF': 2.5,
    'mrna_half_life_Target': 3.7,
    'TF_protein_half_life': 28,
    'Target_protein_half_life': 28,
    'TF_protein_production_rate': 0.059,
    'Target_protein_production_rate': 0.059,
    'n': 2,
    'labeling_efficiency': 1,
    'pulse_time': 60,
    'num_cells': 20_000,
    'dynamics': 'MM',
    'capture_efficiency': 1
}

params_median_extreme_ends = {
    'k_on_TF': 0.13,
    'k_off_TF': 8.4,
    'burst_size_TF': 32,
    'k_on_Target': 0.47,
    'k_off_Target': 7.7,
    'burst_size_Target': 40,
    'splicing_half_life_minutes': 7,
    'mrna_half_life_TF': 2.5,
    'mrna_half_life_Target': 3.7,
    'TF_protein_half_life': 28,
    'Target_protein_half_life': 28,
    'TF_protein_production_rate': 0.059,
    'Target_protein_production_rate': 0.059,
    'n': 2,
    'labeling_efficiency': 1,
    'pulse_time': 60,
    'num_cells': 20_000,
    'dynamics': 'MM',
    'capture_efficiency': 1
}

# def generate_param_args(params_median):
#     ranges_for_kOn = np.linspace(0.125, 0.5, 10)
#     ranges_for_kOff = np.linspace(1.33, 80, 10)
#     ranges_for_mRNA_halflife = np.linspace(1, 6, 10)
#     ranges_for_protein_halflife = np.linspace(10, 90, 10)
#     ranges_for_burst_size = np.linspace(15, 80, 10)
#     ranges_for_translation_rate = np.linspace(0.020, 0.320, 10)

#     # Generate a list of parameter sets
#     list_param_dict = []
#     for key in params_median.keys():
#         if key == 'k_on_TF' or key == 'k_on_Target':
#             for k in ranges_for_kOn:
#                 param_dict = params_median.copy()
#                 param_dict[key] = k
#                 list_param_dict.append(param_dict)
#         elif key == 'k_off_TF' or key == 'k_off_Target':
#             for k in ranges_for_kOff:
#                 param_dict = params_median.copy()
#                 param_dict[key] = k
#                 list_param_dict.append(param_dict)
#         elif key == 'burst_size_TF' or key == 'burst_size_Target':
#             for k in ranges_for_burst_size:
#                 param_dict = params_median.copy()
#                 param_dict[key] = k
#                 list_param_dict.append(param_dict)
#         elif key == 'mrna_half_life_TF' or key == 'mrna_half_life_Target':
#             for k in ranges_for_mRNA_halflife:
#                 param_dict = params_median.copy()
#                 param_dict[key] = k
#                 list_param_dict.append(param_dict)
#         elif key == 'protein_half_life':
#             for k in ranges_for_protein_halflife:
#                 param_dict = params_median.copy()
#                 param_dict[key] = k
#                 list_param_dict.append(param_dict)
#         elif key == 'protein_production_rate' :
#             for k in ranges_for_translation_rate:
#                 param_dict = params_median.copy()
#                 param_dict[key] = k
#                 list_param_dict.append(param_dict)
#     return list_param_dict

def get_param_dict(path_param_csv):
    """
    Reads a CSV file containing parameter sets and returns a list of dictionaries.
    Each dictionary corresponds to a row in the CSV file, with keys as column headers.
    """
    df = pd.read_csv(path_param_csv, index_col=0)
    param_dicts_list = df.to_dict(orient='records')
    return param_dicts_list

def caller(args_list):
    args, filename = args_list
    # all_samples = simulate_asynchronous(**args)
    all_samples = simulate_both_TF(**args)
    # all_samples = simulate(**args)
    # all_samples = simulate_self_regulation(**args)
    all_samples.to_csv(filename)

if __name__ == '__main__':
    #optional: run sim many times
    num_runs = 1
    args_list = []
    #take starting index as arguments
    startIndex = int(sys.argv[1])
    # this_param_quartile_args = params_median.copy()

    
    # param_args_list = [params_median_both_TF]
    # args_list = [(param_args_list[0], os.path.join(output_folder, f"samples_replicates_with_regulation_{startIndex}.csv"))]
    ################################################################################################################################################
    # To generate parameter variations for asynchronous divisions
    # param_args_list = []
    # for variation in np.linspace(0.001, 0.01, 10):
    #     param = params_median.copy()
    #     param['sigma'] = variation
    #     param_args_list.append(param)
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder, exist_ok=True)
    # for index, param_run in enumerate(param_args_list):
    #     if index < startIndex:
    #         continue
    #     args_list.append((param_run, os.path.join(output_folder, f"samples_replicates_with_regulation_{index}.csv")))
    # # loop over list of parameters
    # with multiprocessing.Pool(processes=32) as pool:
    #     pool.map(caller, args_list)
    ################################################################################################################################################
    # To generate parameter variations for two-way regulation
    output_folder = "/home/mzo5929/Keerthana/grnInference/simulationData/large_scale_parameter_scan/two_way_regulation_new/"
    param_csv_path  = "/home/mzo5929/Keerthana/grnInference/simulationData/parameters_25000_both_TF.csv"
    param_args_list = get_param_dict(param_csv_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    for index, param_run in enumerate(param_args_list):
        if index < startIndex:
            continue
        file_path = os.path.join(output_folder, f"samples_replicates_with_regulation_two_way_regulation_{index}.csv")
        # Check if the file already exists
        if os.path.exists(file_path):
            continue
        args_list.append((param_run, file_path))
    # loop over list of parameters
    batch_size = 32
    total_items = len(args_list)
    print(f"Total items to process: {total_items}")
    for start_idx in range(0, total_items, batch_size):
        end_idx = min(start_idx + batch_size, total_items)
        current_batch = args_list[start_idx:end_idx]
        
        
        with multiprocessing.Pool(processes=32) as pool:
            pool.map(caller, current_batch)
    # caller([param_args_list[0], os.path.join(output_folder, f"samples_replicates_with_regulation_{startIndex}.csv")])
###############################################################################################################################################
    #Simulate self regulation
    # output_folder = "/home/mzo5929/Keerthana/grnInference/simulationData/self_regulation/"
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder, exist_ok=True)
    # param_args_list = [params_median_extreme_ends]
    # startIndex = 1
    # caller([param_args_list[0], os.path.join(output_folder, f"A_B_Bloops_{startIndex}.csv")])

    
