#!/bin/bash

CURR=$(pwd)

echo "Formatting Folders for training..."
TRAIN=$CURR/train

cd $TRAIN
for d in */ ;
do
    cd $TRAIN/$d
    rm *.txt
    cd images
    mv * ../
    cd ../
    rmdir images
done

echo "...Complete"
