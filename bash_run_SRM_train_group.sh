#!/bin/bash


# Set up the environment

module load pyger/beta
# set the current dir
currentdir=`pwd`

k1=5
k2=100
python SRM_train_group_bash.py $k1 $k2
