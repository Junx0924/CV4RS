#!/bin/bash

export LD_LIBRARY_PATH=/home/users/p/paka0401/lib
source /home/users/p/paka0401/divide_and_conquer_venv/bin/activate
module load nvidia/cuda/10.0
export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

python /home/users/p/paka0401/CV4RS/CV4RS/2_metric-learning-divide-and-conquer/experiment.py --dataset=mlrsnet --nb-clusters=60 --sz-embedding=4096 --backend=faiss-gpu