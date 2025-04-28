#!/bin/bash
#SBATCH --account=p31666
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=38
#SBATCH --mem=100GB
#SBATCH --time=48:00:00
#SBATCH --job-name=EdgePseudo3
#SBATCH --output=/home/mzo5929/Keerthana/GRNsimulation/highthroughputData/linearNetworkSim/3/logs/slurmLog-%A_%a-%x.out
#SBATCH --error=/home/mzo5929/Keerthana/GRNsimulation/highthroughputData/linearNetworkSim/3/logs/slurmLog-%A_%a-%x.err
#SBATCH --array=0-5

module purge
eval "$(conda shell.bash hook)"
conda activate grnSimulationQuest

startIndex=$((SLURM_ARRAY_TASK_ID *1750))
python /home/mzo5929/Keerthana/GRNsimulation/src/codeGitRepo/src/highThroughputSimulationCode/new/simulationSetup_LinearNetworks.py $startIndex 2
