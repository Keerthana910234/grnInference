#!/bin/bash
#SBATCH --account=p31666
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=100GB
#SBATCH --time=4:00:00
#SBATCH --output=/home/mzo5929/Keerthana/Kuznets-Speck_PerturbationAnalysis/logs/%j-%x.out
#SBATCH --error=/home/mzo5929/Keerthana/Kuznets-Speck_PerturbationAnalysis/logs/%j-%x.err

# # Check if an argument was provided
# if [ $# -eq 0 ]; then
#     echo "Error: No argument provided"
#     exit 1
# fi

# input_number=$1

# module purge
# eval "$(conda shell.bash hook)"
# conda activate grnSimulationQuest

# Code to run the overall python script for the pipeline
# startIndex=$(((SLURM_ARRAY_TASK_ID+1) * 700)-350)
eval "$(conda shell.bash hook)"
conda activate /home/mzo5929/.conda/envs/grnSimulationQuest
which python
startIndex=0
input_number=17
python /home/mzo5929/Keerthana/Kuznets-Speck_PerturbationAnalysis/code/perturbationLinearEffect/perturbationSimulation/simulationSetup_singleGraph.py $startIndex $input_number "g2" 300