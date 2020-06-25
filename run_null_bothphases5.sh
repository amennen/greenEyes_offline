#!/bin/bash
#SBATCH --partition all
#SBATCH -t 4:00:00
#SBATCH --mem-per-cpu=16000
#SBATCH --array=1-100
#SBATCH --job-name null_bothphases
#SBATCH --output new_bothphases/%A_%a.out
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
k2=300
lowhigh=0
classifierType=2
n_iterations=100

python new_permutation_bothphases.py $iter_number $k1 $k2 $lowhigh $classifierType $n_iterations
