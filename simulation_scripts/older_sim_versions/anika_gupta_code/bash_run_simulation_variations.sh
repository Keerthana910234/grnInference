#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=35
#SBATCH --mem=50GB
#SBATCH --time=12:00:00
#SBATCH --job-name=Two-way-regulation
#SBATCH --output=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.out
#SBATCH --error=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.err
#SBATCH --array=0-9

#module purge is to prevent unnecessary modules from interfering with code
module purge
# eval "$(conda shell.bash hook)"
# conda activate grnSimulationQuest
source /home/mzo5929/Keerthana/grnInference/code/.venv/bin/activate
start_index=$((2500 * SLURM_ARRAY_TASK_ID))
# start_index=0
python /home/mzo5929/Keerthana/grnInference/code/grnInferenceRepo/simulationScripts/run_simulator_variations.py $start_index
