#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=21
#SBATCH --mem=5GB
#SBATCH --time=4:00:00
#SBATCH --job-name=figure_3_replicates
#SBATCH --output=/home/gzu5140/Keerthana_b1042/grnInference/logs/slurmLog-%A-%x.out
#SBATCH --error=/home/gzu5140/Keerthana_b1042/grnInference/logs/slurmLog-%A-%x.err

eval "$(conda shell.bash hook)"
conda activate twinfer

python /home/gzu5140/Keerthana_b1042/grnInference/code/grnInferenceRepo/twinfer_simulation_script.py
