#!/bin/bash
# echo "Running experient of $1 to $2 on gpu $3 "
# python pattern_sens_quantized.py --gpu $3 --model Inception --num_bits 8 --qe-mode asym_u --error_pat $1 --des_pat $2 --save_data

echo "Running experiment of $1-$2 on gpu $3"
python error_rate_reduction.py --gpu $3 --model $1 --num_bits 8 --name $1-cifar10-$2-error-rate --method $2 --save_data

# echo "test-case-$1-$2 using gpu $3"
# python fixed_point.py --gpu $3 --model LeNet --num_bits 10 --method test_case --error_pat $1 --des_pat $2 --name LeNet-test-cases-fixed-point-$1-$2 
