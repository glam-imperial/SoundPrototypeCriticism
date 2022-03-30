#!/bin/bash

DATASET_DIR='data_experiment/'
WORKSPACE='experiment_workspace/baseline_cnn/'

DEV_DIR='models_dev'
TEST_DIR='models_test'

ITERATION_MAX=10000


############ Prototype and Criticism Selection  ############
python baseline_cnn/pytorch/adv_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --eps=0.00001 --steps=1 --cuda --isres
