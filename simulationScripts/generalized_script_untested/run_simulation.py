import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import random
import sys
import json
from simulation import simulate
from multiprocessing import Pool
import multiprocessing
from time import sleep

path_to_matrix = "/home/mzo5929/Keerthana/grnInference/simulationData/general_simulation_data/test_data/matrix101.txt"
path_to_param = "/home/mzo5929/Keerthana/grnInference/simulationData/general_simulation_data/test_data/parameter_sheet101.csv"
num_cells = 10
rows = [0,1]
df = simulate(path_to_matrix, path_to_param, num_cells, global_params= None, rows=None)
output_folder = "/home/mzo5929/Keerthana/grnInference/simulationData/general_simulation_data/test_data/"
file_path = os.path.join(output_folder, "simulation_matrix101_parameter_sheet101.csv")
df.to_csv(file_path, index=False)
