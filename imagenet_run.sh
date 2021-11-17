# python compress_classifier.py --arch resnet50 -p 10 -j 22 ~/Datasets/imagenet/ --pretrained --run --qe-config-file ./resnet18_imagenet_post_train.yaml --gpus 2 --name baseline-resnet50-imagenet --method proposed_method --mlc 8 --num_bits 8 --save_data
# python compress_classifier.py --arch densenet121 -p 10 -j 22 ~/Datasets/imagenet/ --pretrained --qe-calibration 0.05 --gpus 2


# DenseNet121
# python compress_classifier.py --arch googlenet -p 10 -j 22 ~/Datasets/imagenet/ --pretrained --qe-calibration 0.05 --gpus 2
# python compress_classifier.py --arch googlenet -p 10 -j 22 /home/imagenet/ --pretrained --run --qe-config-file ./resnet18_imagenet_post_train.yaml --gpus 2 --name baseline-googlenet-imagenet --method proposed_method --mlc 8 --num_bits 8 --save_data

# python compress_classifier.py --arch inception_v3 -p 10 -j 22 ~/Datasets/imagenet/ --pretrained --qe-calibration 0.05 --gpus 2
echo "Running $1-$4 on imangenet using gpu $3 (stat file $2)"
python compress_classifier.py --arch $1 -p 10 -j 22 /home/imagenet/ --pretrained --run --qe-config-file $2 --gpus $3 --name $1-imagenet-$4 --method $4 --mlc 8 --num_bits 8 --save_data

