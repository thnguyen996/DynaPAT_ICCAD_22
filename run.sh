#!/bin/bash
# echo "Running experient of $1 to $2 on gpu $3 "
# python pattern_sens_quantized.py --gpu $3 --model Inception --num_bits 8 --qe-mode asym_u --error_pat $1 --des_pat $2 --save_data

echo "Running experient of $1-$2 on gpu $3"
python quantized_model3.py --gpu $3 --model $1 --num_bits 8 --qe-mode asym_u --name $1-cifar10-$2-2022 --method $2 

# echo "test-case-$1-$2 using gpu $3"
# python quantized_model2.py --gpu $3 --model resnet18 --num_bits 8 --qe-mode asym_u --case $1 --name test-cases-$1-$2 --save_data
