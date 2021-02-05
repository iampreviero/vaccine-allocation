#!/bin/bash

#SBATCH --job-name=Vaccine
#SBATCH --output=out_%a.txt
#SBATCH --error=err_%a.txt
#SBATCH -p sched_mit_sloan_batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=8G

#SBATCH --array=0-80

module load python/3.6.3
module load sloan/python/modules/3.6
xvfb-run -d python3.6 main_cluster.py ${SLURM_ARRAY_TASK_ID} exper3
