# A Comparative Analysis of Multi-Task Learning Approaches in the Context of Multi-Label Remote Sensing Image Retrieval
> 

## Table of contents
* [General info](#general-info)
* [Datasets](#datasets)
* [Requirements](#requirements)
* [Setup](#setup)
* [Training](#training)
* [Evaluation](#evaluation)
* [Implemented Methods](#implemented-methods)
* [Contact](#contact)

## General info
This project aims to compare the performance of multi-task approaches in content-based remote sensing image re-trieval (CBIR). The goal of all the methods in this work isto learn a metric for multi-label images, such that sampleswith maximum overlap in label sets are close. The three multi-task methods we compared are:
1.  Diverse Visual Feature Aggregation for Deep MetricLearning (Diva) [git](https://github.com/Confusezius/ECCV2020_DiVA_MultiFeature_DML)  [pdf](https://arxiv.org/abs/2004.13458)
2.  Divide and Conquer the Embedding Space for MetricLearning (D&C) [git](https://github.com/CompVis/metric-learning-divide-and-conquer)  [pdf](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sanakoyeu_Divide_and_Conquer_the_Embedding_Space_for_Metric_Learning_CVPR_2019_paper.pdf)
3.  Deep Metric Learning with BIER: Boosting Indepen-dent Embedding Robustly (Bier) [git](https://github.com/mop/bier)  [pdf](https://arxiv.org/abs/1801.04815)

One single approach for further comparisons:
1.   Graph Relation Network: Modeling Relations Between Scenes for Multilabel Remote-Sensing Image Classification and Retrieval (SNDL) [pdf](https://elib.dlr.de/137923/1/09173783.pdf)

## Datasets
Data for:
1. [BigEarthNet](http://bigearth.net) 
2. [MLRSNet](https://github.com/cugbrs/MLRSNet)

Downloaded data should be placed in a folder named Dataset and keep the original structure:
```
Dataset
└───BigEarthNet
|    └───S2A_MSIL2A_20170613T101031_0_48
|           │   S2A_MSIL2A_20170613T101031_0_48_B0
|           │   ...
|    ...
|
└───MLRSNet
|    |   Categories_names.xlsx
|    └───Images
|    |      └───airplane
|    |              │   airplane_00001.jpg
|    |              │   ...
|    |
|    └───labels
|    |      └───airplane.csv
|    ...
```
Assuming your folder is placed in e.g. `<$path/Dataset/BigEarthNet>`, pass `$path/Dataset` as input to `--source_path`
## Requirements 
* python==3.6
* torch==1.7.0
* torchvision==0.8.1
* faiss-gpu==1.6.5
* hypia==0.0.3
* GDAL==3.0.4
* pretrainedmodels==0.7.4
* wandb==0.10.20
* vaex==4.0.0
## Setup
An exemplary setup of a virtual environment containing everything needed:
```
(1) wget  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
(2) bash Miniconda3-latest-Linux-x86_64.sh (say yes to append path to bashrc)
(3) source .bashrc
(4) conda create -n DL python=3.6
(5) conda activate DL
(6) conda install matplotlib scipy scikit-learn scikit-image tqdm vaex pillow xlrd
(7) conda install pytorch torchvision faiss-gpu cudatoolkit=10.0 -c pytorch
(8) pip install wandb pretrainedmodels hypia
(9) Run the scripts!
```
## Training
Training is done by using `train_baseline.py` or `train_bier.py` or `train_dac.py` or`train_sndl.py` and setting the respective flags, all of which are listed and explained in `parameters.py`. A set of exemplary runs is provided in `SampleRun.sh`.

**[I.]** **A basic sample run using default parameters would like this**:

```
python train_diva.py --log_online \
                    --dataset MLRSNet  \
                    --source_path ".../Dataset" \
                    --save_path "../Training_Results" \
                    --project MLRSNet --group bier --savename 'bier' \
                    --num_samples_per_class 2  --use_npmem --eval_epoch 10 --nb_epochs 120  
```
#### Some Notes:
* During training, metrics listed in `--eval_metric` will be logged for validation/test set. If you also want to log the overlap of embedding distance from intra and inter group, simply set the flag `--is_log_intra_inter_overlap`. A checkpoint is saved for improvements on recall@1 on train set. The default metrics supported are Recall@K, R-Precision@K, MAP@K.
* If a training is stopped accidentally, you can resume the training by set the flag `--load_from_checkpoint`, the training will be restarted from the last checkpoint epoch, and the training results will be written to the original checkpoint folder.
### Logging results with W&B
* Create an account here (free): https://wandb.ai
* After the account is set, make sure to include your API key in `parameters.py` under `--wandb_key`.
* Set the flag `--log_online` to use wandb logging, if the network is unavailable in your training environment, set the flag `--wandb_dryrun` to make wandb store the data locally, and you can upload the data with the command `wandb sync <$path/wandb/offline..>`
## Evaluation
Evaluation is done by using `evaluate_model.py` and setting the respective flags, all of which are listed and explained in `evaluate_model.py`. A set of exemplary runs is provided in `SampleRun.sh`. The evaluation results will be saved in your checkpoint folder, and will include a summary of metric scores, png files of retrieved samples, distance density plot of intra and inter group if it is evaluated on val set.

## Implemented Methods
### Loss functions
* **Margin loss** [[Sampling Matters in Deep Embeddings Learning](https://arxiv.org/abs/1706.07567)]
* **Binominal loss(boosted)**
* **NCA loss** [[Improving Generalization via Scalable Neighborhood Component Analysis](https://www.umbc.edu/rssipl/people/aplaza/Papers/Journals/2020.TGRS.GRN.pdf)]
* **Fast MOCO** Momentum Contrast Loss
* **Adversarial loss**
### Batch miner
* **Semihard** [[Facenet: A unified embedding for face recognition and clustering](https://arxiv.org/abs/1503.03832)]
* **MultiLabelSemihard** [[A variation of semihard, take embedding vectors and multi-hot labels as input](https://github.com/abarthakur/multilabel-deep-metric)]
* **Distance** [[Sampling Matters in Deep Embeddings Learning](https://arxiv.org/abs/1706.07567)]
### Architectures
* **ResNet50** [[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)]
### Evaluation Metrics
**Metrics based on samples**
* **Recall@K**
* **R-Precision@K**
* **MAP@K**
## Contact
Created by Jun Xiang, email: xj.junxiang@gmail.com - feel free to contact me!