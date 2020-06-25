#!/bin/bash
#SBATCH --partition all
#SBATCH -t 1:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --array=1-1000
#SBATCH --job-name SRM_group_training
#SBATCH --output accuracy_SRM_randomized/SRM_group_training-%A-%a.out
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
k1=20
k2=100
lowhigh=1
python SRM_train_group_randomize_storydiff.py $iter_number $k1 $k2 $lowhigh
