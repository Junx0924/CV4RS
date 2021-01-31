#!/bin/bash
#SBATCH -J dac
#SBATCH -o /home/users/j/jun0924/log/dac.out

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=8
#SBATCH --gres=gpu:tesla:1

#SBATCH --mem=30G
#SBATCH --time=40:00:00
#SBATCH --partition=gpu

#SBATCH --mail-type=ALL
#SBATCH --mail-user=xj.junxiang@gmail.com

source /home/users/j/jun0924/venv/bin/activate
export LD_LIBRARY_PATH=/home/users/j/jun0924/lib
module load nvidia/cuda/10.0
#python /home/users/j/jun0924/CV4RS/train_dac.py  --log_online --dataset BigEarthNet  --project BigEarthNet  --source_path "/scratch/CV4RS/Dataset" --save_path "/home/users/j/jun0924/Training_Results" --group dac_bs50_frozen --dac_nb_clusters 4 --use_hdf5 --batch-size 50 --frozen 
python /home/users/j/jun0924/CV4RS/train_dac.py  --log_online --dataset MLRSNet  --project MLRSNet  --source_path "/scratch/CV4RS/Dataset" --save_path "/home/users/j/jun0924/Training_Results" --group dac_bs50_frozen --dac_nb_clusters 8 --batch-size 50 --frozen 