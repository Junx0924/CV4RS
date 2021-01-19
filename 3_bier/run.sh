TF_LOG_PATH=/media/nax/tensor-bier/products_new_release
mkdir -p ${TF_LOG_PATH}

CUDA_VISIBLE_DEVICES="0" python -u train_bier.py \
    --dataset_name BigEarthNet \
    --source_path /scratch/CV4RS/Dataset\
    --lr-anneal 25000 \
    --lr-decay 0.5 \
    --num-iterations 100000 \
    --labels-per-batch 64 \
    --samples_per_class 2 \
    --batch-size 128 \
    --regularization adversarial \
    --skip-test \
    --lambda-weight 100000.0 \
    --lambda-div 5e-5 \
    --logdir ${TF_LOG_PATH}/stfd-adversarial-bier
