#!/bin/bash
# #SBATCH --partition all
#SBATCH --partition debug
#SBATCH -t 2:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --array=1
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

iter_number=$(($SLURM_ARRAY_TASK_ID - 1))
k1=10
k2=50
lowhigh=0
classifierType=1
filterType=0
n_iterations=1000

python new_data_bothphases-gmcgrath.py $iter_number $k1 $k2 $lowhigh $classifierType $filterType  $n_iterations
