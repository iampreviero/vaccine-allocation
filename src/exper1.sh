#!/bin/bash

#SBATCH --job-name=exper1
#SBATCH --output=exper1_out_%a.txt
#SBATCH --error=exper1_err_%a.txt
#SBATCH -p sched_mit_sloan_batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=8G

#SBATCH --array=0-19

module load python/3.6.3
module load sloan/python/modules/3.6
module load sloan/python/modules/python-3.6/gurobipy/9.0.1
xvfb-run -d python3.6 main_cluster.py ${SLURM_ARRAY_TASK_ID} exper1
