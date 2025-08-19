#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=25
#SBATCH --mem=30GB
#SBATCH --time=4:00:00
#SBATCH --job-name=twinfer_10k_pairs
#SBATCH --output=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.out
#SBATCH --error=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.err

eval "$(conda shell.bash hook)"
conda activate grnSimulationQuest

python /home/mzo5929/Keerthana/grnInference/code/grnInferenceRepo/twinfer_simulation_script.py
