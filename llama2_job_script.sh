#!/bin/sh
# Grid Engine options (lines prefixed with #$)
$ -N llama2_correct
$ -cwd
$ -l h_rt=00:5:00
$ -l h_vmem=5G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem

# Request one GPU in the gpu queue:
$ -q gpu
$ -pe gpu-a100 1
#
# Request 4 GB system RAM
# the total system RAM available to the job is the value specified here multiplied by
# the number of requested GPUs (above)
#$ -l h_vmem=4G

# Initialise the environment modules and load CUDA version 11.0.2
#. /etc/profile.d/modules.sh
#module load cuda

# Run program
python llama2_corrector.py


