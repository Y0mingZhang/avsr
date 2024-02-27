#!/bin/bash

#SBATCH --job-name="disubmit"
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=40GB

srun $@
