#!/bin/bash

#SBATCH -o /home/users/p/paka0401/log.out
#SBATCH -J divide_and_conquer_bigearth_and_mlrsnet
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=15G
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL --mail-user=paul.kaufmann12@gmail.com

module load python/3.7.9
module load nvidia/cuda/10.0

python experiment.py --dataset=bigearth --nb-clusters=43 --backend=faiss-gpu

python experiment.py --dataset=mlrsnet --nb-clusters=60 --backend=faiss-gpu