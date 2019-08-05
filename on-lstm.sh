#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=5-10:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --output=python_array_job_slurm_%j.out
#SBATCH --error=python_array_job_slurm_%j.out

python < /gpfsnyu/home/yz6492/on-lstm/code/main.py

