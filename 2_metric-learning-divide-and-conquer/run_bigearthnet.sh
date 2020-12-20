#!/bin/bash

#SBATCH -o /home/users/p/paka0401/CV4RS/CV4RS/2_metric-learning-divide-and-conquer/log/bigearth_log.out
#SBATCH -J D&C_bigearth
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=15G
#SBATCH --partition=gpu
#SBATCH --time=30:00:00
#SBATCH --mail-type=ALL --mail-user=paul.kaufmann12@gmail.com

export LD_LIBRARY_PATH=/home/users/p/paka0401/lib
source /home/users/p/paka0401/divide_and_conquer_venv/bin/activate
module load python/3.7.1
module load nvidia/cuda/10.0
export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

python /home/users/p/paka0401/CV4RS/CV4RS/2_metric-learning-divide-and-conquer/experiment.py --dataset=bigearth --nb-clusters=43 --sz-embedding=128 --backend=faiss-gpu