#!/bin/bash
# Heavily based on code created by Dr. Andrew Fagg
# Modified: Stuart Edris
#1;95;0c
# Run a set of ML experiments to identify FD
#  The --array line says that we will execute N experiments, N being the number of folds 
#  (and thus rotations) in the reanalysis model.
#   You can specify ranges or comma-separated lists on this line
#  For each experiment, the SLURM_ARRAY_TASK_ID will be set to the experiment number
#   In this case, this ID is used to set the name of the stdout/stderr file names
#   and is passed as an argument to the python program
# 
# Reasonable partitions: debug_5min, debug_30min, normal/32gb_20core/64gb_24core
#  
# Note: the normal partition combines the 32gb and 64gb cores.
# Current hypothesis is the numpy.core._exceptions._ArrayMemoryError that sometimes
# occurrs is on the 32gb cores. 64gb cores are specified to avoid this.
# The memory error says it is unable to allocate ~300 MiB.
#
# ML methods to explore: rf, svm, ada, ann, cnn, rnn, transformer, gans
# Number of locations to change ML name: 4 (2 output filenames, 1 experiment, 1 in Python command (.txt file)
# From observation: rf uses ~24 gb of memory, needs ~30 hr
# From observation: Ada uses ~22 gb of memory, needs ~20 hr
# From observation: SVMs uses ~30.6 gb of memory, needs ~6 hr
# From observation: ANNs uses ~25 gb of memory, needs ~36 hr
# From observation: RNNs uses ~30 gb of memory, needs ~19 hr
# From observation: LSTMs uses ~29 gb of memory, needs ~23 hr
# 

#SBATCH --partition=64gb_24core
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=34816
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID. Note this needs to be changed with method, reanalaysis model, and ML model
##SBATCH --exclusive
# Request to use the notes exclusively for the task or other ML tasks if the memory is available (TF likes to claim all available memory)
#SBATCH --output=Data/outputs_christian/era5_christian_LSTM_exp%04a_stdout.txt
#SBATCH --error=Data/outputs_christian/era5_christian_LSTM_exp%04a_stderr.txt
#SBATCH --time=47:00:00
#SBATCH --job-name=era5_christian_LSTM
#SBATCH --mail-user=sgedris@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/rarrell
#SBATCH --array=0-42
#
#################################################



### Code to run on the supercomputer
. /home/rarrell/tf_setup.sh
conda activate tf

# Note only one set of experiments should be uncommented at per batch run
# After each batch run, run the same python ML_and_FD.py command with --check
hostname

# Methods to run this for:
# christian
# nogeura
# pendergrass
# liu
# otkin
# 4 cases where the method name needs to be changed (2 for output file names, 1 for experiment name, 1 for the python command)

# Perform the experiments
python Scripts/ML_and_FD.py @Scripts/christian/era5_LSTM_parameters.txt --ra_model era5 --globe --method christian --rotation $SLURM_ARRAY_TASK_ID --ntrain_folds 20 --dataset /scratch/rarrell/Data -vv 

# Create several figures
# This line runs at the end for either local or supercomputer runs
# Run this after the ML experiments have been run for all methods
# python Scripts/ML_and_FD.py @Scripts/rf_parameters.txt --ra_model nldas --method liu --ntrain_folds 41 --evaluate --nogo
