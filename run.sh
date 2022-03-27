#!/bin/bash

DATASET_DIR='data_experiment/'
WORKSPACE='experiment_workspace/baseline_cnn/'

DEV_DIR='models_dev'
TEST_DIR='models_test'

ITERATION_MAX=10000

############ Preprocessing ##########
python preprocessing/preprocessing.py
python preprocessing/data_split.py

