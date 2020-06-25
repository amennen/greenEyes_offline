#!/bin/bash
#SBATCH --partition all
#SBATCH -t 5:00:00
#SBATCH --mem-per-cpu=16000
#SBATCH --job-name SRM_group_training
#SBATCH --output new_group_SRM/SRM_phase1-%A.out
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=amennen@princeton.edu

echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"

# Set up the environment

module load pyger/beta
# set the current dir
currentdir=`pwd`


k1=1
python new_phase1_group_training.py $k1
