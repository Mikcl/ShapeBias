#!/bin/bash

CURR=$(pwd)

echo "Formatting Folders for validation"
VAL=$CURR/val

cd $VAL

while IFS= read -r line; do
    # OUTPUT="$(ls -1)"
    # echo "${OUTPUT}"
    image=$(echo "$line" | awk '{print $1;}')
    class=$(echo "$line" | awk '{print $2}')

    mkdir -p $class
    mv ./images/$image ./$class
done < "$VAL/val_annotations.txt"

rmdir images
rm ./val_annotations.txt

echo "....Complete"