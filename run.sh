#!/bin/bash
echo "Training the model"
python3 /train.py -model 1
echo "Generating predictions"
python3 /predict.py -model 1
