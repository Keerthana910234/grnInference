#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=33
#SBATCH --mem=10GB
#SBATCH --time=12:00:00
#SBATCH --job-name=A_to_B
#SBATCH --output=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.out
#SBATCH --error=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.err

source /home/mzo5929/Keerthana/grnInference/code/.venv/bin/activate
python /home/mzo5929/Keerthana/grnInference/code/grnInferenceRepo/simulation_scripts/gillespie_simulation/gillespie_script_copy_copy.py

