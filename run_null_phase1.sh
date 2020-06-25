#!/bin/bash
#SBATCH --partition all
#SBATCH -t 5:00:00
#SBATCH --mem-per-cpu=16000
#SBATCH --array=1-1000
#SBATCH --job-name phase1_null
#SBATCH --output new_group_SRM/%A-%a.out
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=amennen@princeton.edu

echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"

# Set up the environment

module load pyger/beta
# set the current dir
currentdir=`pwd`

iter_number=$(($SLURM_ARRAY_TASK_ID - 1))
k1=0

python new_permutation_phase1.py $k1 $iter_number
