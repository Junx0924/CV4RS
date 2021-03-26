source /home/users/j/.../venv/bin/activate
export LD_LIBRARY_PATH=/home/users/.../lib
module load nvidia/cuda/10.0
### for training
python ./train_bier.py  --log_online --dataset MLRSNet  --project MLRSNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results" --savename 'bier' --group bier  --num_samples_per_class 2 --frozen  --use_npmem --eval_epoch 10 --nb_epochs 120  
python ./train_dac.py  --log_online --dataset MLRSNet  --project MLRSNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results" --savename 'dac' --group dac --dac_nb_clusters 8 --num_samples_per_class 2 --frozen  --use_npmem --eval_epoch 10 --nb_epochs 120 
python ./train_diva.py  --log_online --dataset MLRSNet  --project MLRSNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results"  --savename 'diva' --group diva --num_samples_per_class 2 --use_npmem --frozen   -eval_epoch 10 --nb_epochs 120 
python ./train_baseline.py  --log_online --dataset MLRSNet  --project MLRSNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results"  --savename 'baseline' --group baseline --num_samples_per_class 2 --use_npmem --frozen   --eval_epoch 10 --nb_epochs 120  

python ./train_bier.py  --log_online --dataset BigEarthNet  --project BigEarthNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results"  --savename 'bier' --group bier  --num_samples_per_class 2 --frozen  --use_npmem --eval_epoch 10 --nb_epochs 120  
python ./train_dac.py  --log_online --dataset BigEarthNet  --project BigEarthNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results"  --savename 'dac' --group dac --dac_nb_clusters 8 --num_samples_per_class 2 --frozen  --use_npmem --eval_epoch 10 --nb_epochs 120 
python ./train_diva.py  --log_online --dataset BigEarthNet  --project BigEarthNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results"  --savename 'diva' --group diva --num_samples_per_class 2 --use_npmem --frozen    --eval_epoch 10 --nb_epochs 120
python ./train_baseline.py  --log_online --dataset BigEarthNet  --project BigEarthNet  --source_path "/scratch/CV4RS/Dataset" --save_path "../Training_Results"  --savename 'baseline' --group baseline --num_samples_per_class 2 --use_npmem --frozen    --eval_epoch 10 --nb_epochs 120  

### for evaluation
python ./evaluate_model.py --load_from_checkpoint "../Training_Results/MLRSNet/dac" --source_path "/scratch/CV4RS/Dataset"
python ./evaluate_model.py --load_from_checkpoint "../Training_Results/BigEarthNet/dac" --source_path "/scratch/CV4RS/Dataset"
python ./evaluate_model.py --load_from_checkpoint "../Training_Results/MLRSNet/baseline" --source_path "/scratch/CV4RS/Dataset"
python ./evaluate_model.py --load_from_checkpoint "../Training_Results/BigEarthNet/baseline" --source_path "/scratch/CV4RS/Dataset"
python ./evaluate_model.py --load_from_checkpoint "../Training_Results/MLRSNet/bier" --source_path "/scratch/CV4RS/Dataset"
python ./evaluate_model.py --load_from_checkpoint "../Training_Results/BigEarthNet/bier" --source_path "/scratch/CV4RS/Dataset"
python ./evaluate_model.py --load_from_checkpoint "../Training_Results/MLRSNet/diva" --source_path "/scratch/CV4RS/Dataset"
python ./evaluate_model.py --load_from_checkpoint "../Training_Results/BigEarthNet/diva" --source_path "/scratch/CV4RS/Dataset"