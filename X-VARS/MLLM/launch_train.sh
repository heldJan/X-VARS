#!/bin/bash
#SBATCH --output=test2_mv.txt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem-per-gpu=64G	 # Memory to allocate in MB per allocated CPU core
#SBATCH --time="0-04:05:00"	 # Max execution time

#SBATCH --mail-user=jan.held@student.uliege.be
#SBATCH --account=telim

# Activate the Anaconda environment in which to execute the Jupyter instance.
#/home/jheld/anaconda3
# C:\Users\Jan Held\anaconda3\etc\profile.d\conda.sh
source /gpfs/home/acad/ulg-intelsig/jheld/anaconda3/etc/profile.d/conda.sh
# conda activate pytorch

conda activate vars-ex

accelerate launch --config_file /gpfs/home/acad/ulg-intelsig/jheld/.cache/huggingface/accelerate/default_config.yaml training.py
