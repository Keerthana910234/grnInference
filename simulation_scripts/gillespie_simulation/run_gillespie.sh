#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=34
#SBATCH --mem=10GB
#SBATCH --time=8:00:00
#SBATCH --job-name=A_to_B_r_add_dependent
#SBATCH --output=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.out
#SBATCH --error=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.err

source /projects/b1042/GoyalLab/Keerthana/grnInference/code/.venv/bin/activate
start_index=105 #$((65 * SLURM_ARRAY_TASK_ID))
path_to_parameter="/projects/b1042/GoyalLab/Keerthana/grnInference/simulation_data/gillespie_simulation_test/sim_details/effect_of_r_add_sampling_dependent.csv"
path_to_interaction_matrix="/projects/b1042/GoyalLab/Keerthana/grnInference/simulation_data/gillespie_simulation_test/sim_details/interaction_matrix_positive.txt"
path_to_output_folder="/projects/b1042/GoyalLab/Keerthana/grnInference/simulation_data/gillespie_simulation_test/A_to_B_r_add_dependent/"
path_to_log_file="/projects/b1042/GoyalLab/Keerthana/grnInference/simulation_data/gillespie_simulation_test/sim_details/A_and_B.jsonl"
type_of_interaction="A_to_B"

# Run Python script with matching CLI arguments
python /projects/b1042/GoyalLab/Keerthana/grnInference/code/grnInferenceRepo/simulation_scripts/gillespie_simulation/gillespie_script.py \
    --matrix_path "$path_to_interaction_matrix" \
    --param_csv "$path_to_parameter" \
    --row_to_start "$start_index"\
    --output_folder "$path_to_output_folder"\
    --log_file "$path_to_log_file" \
    --type "$type_of_interaction"