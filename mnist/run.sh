#!/bin/bash

#SBATCH --job-name="mnist"
#SBATCH -D .
#SBATCH --output=output/output.out
#SBATCH --error=error/error.err
#SBATCH --nodes=1

source activate pytorch

python train.py
