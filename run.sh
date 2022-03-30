#!/bin/bash

DATASET_DIR='data_experiment/'
WORKSPACE='experiment_workspace/baseline_cnn/'

DEV_DIR='models_dev'
TEST_DIR='models_test'

ITERATION_MAX=10000

############ Preprocessing ##########
python preprocessing/preprocessing.py
python preprocessing/data_split.py


############ Original DNNs ###########

# CNN-8
## Development ##
python baseline_cnn/pytorch/main_org.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda
python baseline_cnn/pytorch/main_org.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda
## Test ##
python baseline_cnn/pytorch/main_org.py train --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda
python baseline_cnn/pytorch/main_org.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda


# ResNet
## Development ##
python baseline_cnn/pytorch/main_org.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda --isres
python baseline_cnn/pytorch/main_org.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda --isres
## Test ##
python baseline_cnn/pytorch/main_org.py train --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda --isres
python baseline_cnn/pytorch/main_org.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda --isres


############ Original DNNs + Attention ######

# CNN-8_Att
## Development ##
python baseline_cnn/pytorch/main_att.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda
python baseline_cnn/pytorch/main_att.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda
## Test ##
python baseline_cnn/pytorch/main_att.py train --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda
python baseline_cnn/pytorch/main_att.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda

# ResNet_Att
## Development ##
python baseline_cnn/pytorch/main_att.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda --isres
python baseline_cnn/pytorch/main_att.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda --isres
## Test ##
python baseline_cnn/pytorch/main_att.py train --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda --isres
python baseline_cnn/pytorch/main_att.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda --isres


############ Dilated DNNs + Attention ######

# CNN-8_Dila_Att
## Development ##
CUDA_VISIBLE_DEVICES=2, python baseline_cnn/pytorch/main_dia_att.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda
CUDA_VISIBLE_DEVICES=2, python baseline_cnn/pytorch/main_dia_att.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda
## Test ##
python baseline_cnn/pytorch/main_dia_att.py train --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda
python baseline_cnn/pytorch/main_dia_att.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda

# ResNet_Dila_Att
## Development ##
python baseline_cnn/pytorch/main_dia_att.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda --isres
python baseline_cnn/pytorch/main_dia_att.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda --isres
## Test ##
python baseline_cnn/pytorch/main_dia_att.py train --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda --isres
python baseline_cnn/pytorch/main_dia_att.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda --isres

