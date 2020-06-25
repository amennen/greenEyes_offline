#!/bin/bash
#SBATCH --partition all
#SBATCH -t 30:00:00
#SBATCH --mem-per-cpu=5000
#SBATCH --array=97
#SBATCH --job-name data_bothphases
#SBATCH --output new_bothphases/%A_%a.out
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=amennen@princeton.edu

echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"

# Set up the environment

#module load pyger/beta
conda activate rtAtten
# set the current dir
currentdir=`pwd`

iter_number=$(($SLURM_ARRAY_TASK_ID - 0))
n_iterations=1000

# updating 3/9/20 to test for thesis - final stations for plot
python train_test_stations_bootstrap.py $iter_number $n_iterations
#python train_test_stations_bootstrap_predict_proba.py $iter_number $n_iterations
#python new_data_bothphases_log_array.py $iter_number $n_iterations $classifierType 