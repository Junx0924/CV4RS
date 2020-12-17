#!/bin/bash

#SBATCH -o /home/users/p/paka0401/mlrsnet_log.out
#SBATCH -J divide_and_conquer_mlrsnet
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=15G
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL --mail-user=paul.kaufmann12@gmail.com

module load python/3.7.1
module load nvidia/cuda/10.0
export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1
source divide_and_conquer_venv/bin/activate

python experiment.py --dataset=mlrsnet --nb-clusters=60 --sz-embedding=4096 --backend=faiss-gpu