#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=35
#SBATCH --mem=10GB
#SBATCH --time=2:00:00
#SBATCH --job-name=Parameter-Analysis
#SBATCH --output=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.out
#SBATCH --error=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.err


module purge
# eval "$(conda shell.bash hook)"
# conda activate grnSimulationQuest
source /home/mzo5929/Keerthana/grnInference/code/.venv/bin/activate
# start_index=$((5000 * SLURM_ARRAY_TASK_ID))
start_index=0



# python ./calculate_correlations_high_throughput.py \
#   --path_to_simulations /home/mzo5929/Keerthana/grnInference/simulation_data/gillespie_simulation/A_to_B/ \
#   --output /home/mzo5929/Keerthana/grnInference/analysisData/gillespie_simulation_analysis/A_to_B/ \
#   --genes gene_1_mRNA gene_2_mRNA \
#   --timepoints 5 10 15 20

python ./calculate_correlations_high_throughput.py \
  --path_to_simulations /home/mzo5929/Keerthana/grnInference/simulation_data/gillespie_simulation/A_and_B/ \
  --output /home/mzo5929/Keerthana/grnInference/analysisData/gillespie_simulation_analysis/A_and_B/ \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 5 10 15 20

python ./calculate_correlations_high_throughput.py \
  --path_to_simulations /home/mzo5929/Keerthana/grnInference/simulation_data/gillespie_simulation/A_no_reg_B/ \
  --output /home/mzo5929/Keerthana/grnInference/analysisData/gillespie_simulation_analysis/A_no_reg_B/ \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 5 10 15 20
