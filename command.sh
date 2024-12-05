#!/bin/bash


#In no particular order but use based on intuition which preprocessing data --> training model (make sure to put targets.csv in format as in example which is the y of the training molecules in ~./data) --
# --> computing descriptors for new molecules after running python modify.py and moving file to right place ( see creating_xyz_files) for more details --
# --> predicting properties of new molecule.
## --------------------------------- first make sure the multi-mol xyz of training data (xyz) is done in right form ---------------------###
cd creating_the_xyz/
python3 modify.py train_example.xyz train.xyz    
cp train.xyz ../data/training_xyz_files/train.xyz
cd ..

### --------------------------------- preprocessing training xyz into descriptors gotten from RDKit ------------------------------------- ###
python3 opac3/scripts/preprocess_data.py \
    --input-dir data/training_xyz_files/ \
    --targets-file data/targets.csv \
    --output-descriptors data/descriptors.csv

### ---------------------------------- train a vae model ------------------------------------ ###
python3 opac3/scripts/train_model.py \
    --descriptors-file data/descriptors.csv \
    --targets-file data/targets.csv \
    --model-output models/trained_model.pth \
    --epochs 200 \
    --validation-size 0.2 \
    --learning-rate 0.001 \
    --batch-size 64 \
    --hidden-dim 512 \
    --weight-decay 1e-4

## ------------------------------------- get new descriptors of test molecules in right xyz format ---------------------------##
cd creating_the_xyz/
python3 modify.py test_example.xyz test.xyz
cp test.xyz ../data/testing_xyz_files/test.xyz
cd ..

## --------------------------------------- convert the test xyz to the descriptors recognized by model------------------------##
python3 opac3/scripts/compute_descriptors.py \
    --input-dir data/testing_xyz_files/ \
    --output-descriptors data/new_descriptors.csv

## ------------------------------------------ make predictions of the new molecules ----------------------------------------- ##
python3 opac3/scripts/predict_properties.py \
    --model-file models/trained_model.pth \
    --descriptors-file data/new_descriptors.csv \
    --predictions-output data/predictions.csv

## If you want to do Active Learning ##

## ---------------------------- running active learning ---------------------------- ##
python3 opac3/active_learning/active_learning.py \
    --descriptors-file data/descriptors.csv \
    --targets-file data/targets.csv \
    --initial-train-size 10 \
    --query-size 5 \
    --iterations 2 \
    --model-output models/al_trained_model.pth \
    --hidden-dim 256 \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --weight-decay 1e-4

## ---------------------------predict the properties of new molecules with the AL model -----------------------##
python3 opac3/active_learning/predict_new_data.py \
    --model-file models/al_trained_model.pth \
    --descriptors-file data/new_descriptors.csv \
    --predictions-output data/new_predictions.csv


