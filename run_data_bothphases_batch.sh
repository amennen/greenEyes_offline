#!/bin/bash
#SBATCH --partition all
#SBATCH -t 1:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --array=1-1000
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
k1=0
k2=0
lowhigh=0
classifierType=1
filterType=0

python new_data_bothphases_batch.py $iter_number $k1 $k2 $lowhigh $classifierType $filterType
