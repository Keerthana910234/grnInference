#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=33
#SBATCH --mem=10GB
#SBATCH --time=48:00:00
#SBATCH --job-name=A_to_B
#SBATCH --output=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.out
#SBATCH --error=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.err
#SBATCH --array=0-39

source /projects/b1042/GoyalLab/Keerthana/grnInference/code/.venv/bin/activate
start_index=$((600 * SLURM_ARRAY_TASK_ID))
path_to_parameter="/projects/b1042/GoyalLab/Keerthana/grnInference/simulation_data/parameter_scan_simulations/simulation_details/parameters_3genes_positive_reg_pi_on_r_add_scaled.csv"
path_to_interaction_matrix="/projects/b1042/GoyalLab/Keerthana/grnInference/simulation_data/parameter_scan_simulations/simulation_details/interaction_matrix_A_to_B.txt"
path_to_output_folder="/projects/b1042/GoyalLab/Keerthana/grnInference/simulation_data/parameter_scan_simulations/A_to_B/"
path_to_log_file="/projects/b1042/GoyalLab/Keerthana/grnInference/simulation_data/parameter_scan_simulations/simulation_details/A_to_B.jsonl"
type_of_interaction="A_to_B"

# Run Python script with matching CLI arguments
python /projects/b1042/GoyalLab/Keerthana/grnInference/code/grnInferenceRepo/simulation_scripts/gillespie_simulation/gillespie_script.py \
    --matrix_path "$path_to_interaction_matrix" \
    --param_csv "$path_to_parameter" \
    --row_to_start "$start_index"\
    --output_folder "$path_to_output_folder"\
    --log_file "$path_to_log_file" \
    --type "$type_of_interaction"