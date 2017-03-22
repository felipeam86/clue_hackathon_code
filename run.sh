#!/bin/bash
echo "starting"
python3 /train.py -model 1 -weights weights/lstm_1_layer.hdf5
python3 /predict.py -model 1 -weights weights/lstm_1_layer.hdf5