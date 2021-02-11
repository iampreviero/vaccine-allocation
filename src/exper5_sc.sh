#!/bin/bash

#SBATCH --job-name=exper5
#SBATCH --output=exper5_out_%a.txt
#SBATCH --error=exper5_err_%a.txt
#SBATCH -p normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

#SBATCH --array=0-99

module load anaconda/2020a
module load gurobi/gurobi-903
xvfb-run -d python3.6 main_cluster.py ${SLURM_ARRAY_TASK_ID} exper5
