#!/bin/bash

#SBATCH --job-name="cf-100-base"
#SBATCH -D .
#SBATCH --output=output/output.out
#SBATCH --error=error/error.err
#SBATCH --nodes=1

source activate pytorch

python train.py
