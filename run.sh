#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=<y.chang20@imperial.ac.uk>
export PATH=/vol/bitbucket/yc7020/miniconda3/bin/:$PATH
source activate
source /vol/cuda/11.3.1-cudnn8.2.1/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

DATASET_DIR='data_experiment/'
WORKSPACE='experiment_workspace/baseline_cnn/'

DEV_DIR='models_dev'
TEST_DIR='models_test'

ITERATION_MAX=10000

############ Preprocessing ##########
#python3 preprocessing/preprocessing.py
#python3 preprocessing/data_split.py

############ Development ############
# Train model
python3 baseline_cnn/pytorch/main_org.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda

# Train caanet_model
# python3 baseline_cnn/pytorch/main_caanet.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda

# Plot the att
# python3 baseline_cnn/pytorch/att_caanet.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda

# TSNE
# python3 baseline_cnn/pytorch/tsne.py tsne --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate

# Evaluate
# python3 baseline_cnn/pytorch/adv_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda --eps=0.01 --steps=1

#python3 baseline_cnn/pytorch/adv_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda --eps=0.01 --steps=3

#python3 baseline_cnn/pytorch/adv_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda --eps=0.01 --steps=2

#python3 baseline_cnn/pytorch/adv_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda --eps=0.01 --steps=1

#python3 baseline_cnn/pytorch/adv_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda --eps=0.01 --steps=6

#python3 baseline_cnn/pytorch/adv_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda --eps=0.01 --steps=5

############ Test ############
# Train model
# python3 baseline_cnn/pytorch/main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda

# Evaluate
# python3 baseline_cnn/pytorch/main_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda
