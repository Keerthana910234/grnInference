#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=34
#SBATCH --mem=30GB
#SBATCH --time=48:00:00
#SBATCH --job-name=A_to_B
#SBATCH --output=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.out
#SBATCH --error=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.err
#SBATCH --array=0-30

module purge
# eval "$(conda shell.bash hook)"
# conda activate grnSimulationQuest
source /home/mzo5929/Keerthana/grnInference/code/.venv/bin/activate
start_index=$((750 * SLURM_ARRAY_TASK_ID))
path_to_parameter="/home/mzo5929/Keerthana/grnInference/simulation_data/gillespie_simulation/sim_details/lhc_sampled_parameters_positive_reg_2.csv"
path_to_interaction_matrix="/home/mzo5929/Keerthana/grnInference/simulation_data/gillespie_simulation/sim_details/interaction_matrix_positive.txt"

# Run Python script with matching CLI arguments
python /home/mzo5929/Keerthana/grnInference/code/grnInferenceRepo/simulationScripts/gillespie_simulation/gillespie_script.py \
    --matrix_path "$path_to_interaction_matrix" \
    --param_csv "$path_to_parameter" \
    --row_to_start "$start_index"\
    --output_folder "/home/mzo5929/Keerthana/grnInference/simulation_data/gillespie_simulation/A_to_B/"\
    --log_file "/home/mzo5929/Keerthana/grnInference/simulation_data/gillespie_simulation/sim_details/A_to_B_reg.jsonl" \
    --type "A_to_B"