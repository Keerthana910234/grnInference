#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=35
#SBATCH --mem=30GB
#SBATCH --time=1:00:00
#SBATCH --job-name=No-regulation
#SBATCH --output=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.out
#SBATCH --error=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.err

module purge
# eval "$(conda shell.bash hook)"
# conda activate grnSimulationQuest
source /home/mzo5929/Keerthana/grnInference/code/.venv/bin/activate
# start_index=$((5000 * SLURM_ARRAY_TASK_ID))
start_index=9
python /home/mzo5929/Keerthana/grnInference/code/stochastic-regulation-code/run_simulator.py $start_index
