#!/bin/bash

chmod +x ./* 

CURR=$(pwd)

./data_prep.sh

cd $CURR/tiny-imagenet-200

../train_prep.sh
../val_prep.sh