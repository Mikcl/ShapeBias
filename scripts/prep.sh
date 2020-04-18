#!/bin/bash

chmod +x ./* 

CURR=$(pwd)

./scripts/data_prep.sh

cd $CURR/tiny-imagenet-200

../scripts/train_prep.sh
../scripts/val_prep.sh