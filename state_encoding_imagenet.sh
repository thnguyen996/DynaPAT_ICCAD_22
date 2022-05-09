#!/usr/bin/bash
echo "Running experiment of $1-$2 on gpu $3"
python state_encoding_imagenet.py ~/Datasets/imagenet/val/ --model $1 --pretrained --gpu $3 --method $2 --num_bits 8 --name $1-imagenet-$2 
