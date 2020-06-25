#!/bin/bash
#SBATCH --partition all
#SBATCH -t 30:00:00
#SBATCH --mem-per-cpu=5000
#SBATCH --array=2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,59,62,65,68
#SBATCH --job-name data_bothphases
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

iter_number=$(($SLURM_ARRAY_TASK_ID - 0))
lowhigh=0
classifierType=2
n_iterations=1000
mask=2

python new_data_bothphases_log_array.py $iter_number $lowhigh $classifierType $n_iterations $mask
