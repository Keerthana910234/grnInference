#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=11
#SBATCH --mem=5GB
#SBATCH --time=1:00:00
#SBATCH --job-name=A_to_B_neg_reg
#SBATCH --output=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.out
#SBATCH --error=/home/mzo5929/Keerthana/grnInference/logs/slurmLog-%A-%x.err

source /home/mzo5929/Keerthana/grnInference/code/.venv/bin/activate
python /home/mzo5929/Keerthana/grnInference/code/grnInferenceRepo/simulation_scripts/gillespie_simulation/gillespie_script_copy_copy.py