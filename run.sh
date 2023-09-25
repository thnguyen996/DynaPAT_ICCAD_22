#!/bin/bash
echo "Running experiment of $1-$2 on gpu $3"
python3 main.py --method $2 --model $1 --gran layer --num_bits 8 --gpu $3 --name $1-cifar10-$2-filter --save_data

# echo "Running experiment of $1-$2 on gpu $3"
# python fixed_point.py --gpu $3 --model $1 --num_bits 8 --name $1-cifar10-$2-error-rate --method $2 --save_data

# echo "test-case-$1-$2 using gpu $3"
# python fixed_point.py --gpu $3 --model LeNet --num_bits 10 --method test_case --error_pat $1 --des_pat $2 --name LeNet-test-cases-fixed-point-$1-$2 
