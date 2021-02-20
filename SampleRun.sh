source /home/users/j/jun0924/venv/bin/activate
export LD_LIBRARY_PATH=/home/users/j/jun0924/lib
module load nvidia/cuda/10.0
### for training
python ./train_bier.py  --log_online --dataset MLRSNet  --project MLRSNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results" --group bier  --num_samples_per_class 2 --frozen  --use_hdf5 --eval_epoch 10 --nb_epochs 120
python ./train_dac.py  --log_online --dataset MLRSNet  --project MLRSNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results" --group dac --dac_nb_clusters 8 --num_samples_per_class 2 --frozen  --use_hdf5 --eval_epoch 10 --nb_epochs 120
python ./train_diva.py  --log_online --dataset MLRSNet  --project MLRSNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results" --group diva --num_samples_per_class 2 --use_hdf5 --frozen  --savename combin -eval_epoch 10 --nb_epochs 120
python ./train_snca.py  --log_online --dataset MLRSNet  --project MLRSNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results" --group snca  --batch_size 50  --frozen   --use_hdf5 -eval_epoch 10 --nb_epochs 120

python ./train_bier.py  --log_online --dataset BigEarthNet  --project BigEarthNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results" --group bier  --num_samples_per_class 2 --frozen  --use_hdf5 --eval_epoch 10 --nb_epochs 120
python ./train_dac.py  --log_online --dataset BigEarthNet  --project BigEarthNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results" --group dac --dac_nb_clusters 8 --num_samples_per_class 2 --frozen  --use_hdf5 --eval_epoch 10 --nb_epochs 120
python ./train_diva.py  --log_online --dataset BigEarthNet  --project BigEarthNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results" --group diva --num_samples_per_class 2 --use_hdf5 --frozen  --savename combin -eval_epoch 10 --nb_epochs 120
python ./train_snca.py  --log_online --dataset BigEarthNet  --project BigEarthNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results" --group snca  --batch_size 120  --frozen   --use_hdf5 -eval_epoch 10 --nb_epochs 120

### for evaluation
python ./evaluation.py --load_from_checkpoint "../Training_Results/MLRSNet/dac_s0" --source_path "/scratch/CV4RS/Dataset"
python ./evaluation.py --load_from_checkpoint "../Training_Results/BigEarthNet/dac_s0" --source_path "/scratch/CV4RS/Dataset"
python ./evaluation.py --load_from_checkpoint "../Training_Results/MLRSNet/snca_s0" --source_path "/scratch/CV4RS/Dataset"
python ./evaluation.py --load_from_checkpoint "../Training_Results/BigEarthNet/snca_s0" --source_path "/scratch/CV4RS/Dataset"
python ./evaluation.py --load_from_checkpoint "../Training_Results/MLRSNet/bier_s0" --source_path "/scratch/CV4RS/Dataset"
python ./evaluation.py --load_from_checkpoint "../Training_Results/BigEarthNet/bier_s0" --source_path "/scratch/CV4RS/Dataset"
python ./evaluation.py --load_from_checkpoint "../Training_Results/MLRSNet/combin" --source_path "/scratch/CV4RS/Dataset"
python ./evaluation.py --load_from_checkpoint "../Training_Results/BigEarthNet/combin" --source_path "/scratch/CV4RS/Dataset"