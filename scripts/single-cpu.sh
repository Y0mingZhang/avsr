#!/bin/bash

#SBATCH --job-name="single-cpu"
#SBATCH --partition=cpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB

srun $@
