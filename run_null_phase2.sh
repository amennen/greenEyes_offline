#!/bin/bash
#SBATCH --partition all
#SBATCH -t 1:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --array=1-1000000
#SBATCH --job-name SRM_phase2
#SBATCH --output new_accuracy_phase2/SRM_phase2-%A-%a.out
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

# every thousand: different rep number from phase 1
N=$(($SLURM_ARRAY_TASK_ID - 1))
permuation=$(($N/1000))
iter_number=$((N%1000))
k1=1
k2=0
lowhigh=0 # 0 = all TRs, 1 = low TRs, 2 = high TRs
classifierType=1 # 1 = temporal, 2 = spatiotemporal

python new_phase2_classifying.py $iter_number $k1 $k2 $lowhigh $classifierType $permutation
