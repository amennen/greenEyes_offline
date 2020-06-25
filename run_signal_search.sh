#!/bin/bash
#SBATCH --partition all
#SBATCH -t 25:00:00
#SBATCH --mem-per-cpu=5000
#SBATCH --array=1-132
#SBATCH --job-name signal_search
#SBATCH --output classifier_hunt/%A_%a.out
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
n_iterations=1000
takeAvg=1

python new_correlate_signal_search.py $iter_number $n_iterations $takeAvg
