#!/bin/bash

echo "Retireving Tiny Image Net data"

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip

echo "Unzipping..."

unzip -q ./tiny-imagenet-200.zip

echo "..."

rm ./tiny-imagenet-200.zip

echo "Complete"
