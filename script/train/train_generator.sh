#!/bin/sh

# Pretrain generator

data_dir=/u02/phatth1/dataset
root_exp_dir=/u02/phatth1/runs

# python3 train.py --real_image_dir $data_dir/train_photo \
#                  --anime_image_dir $data_dir/Hayao \
#                  --amp --cache --model v1 --num_workers 4 \
#                  --exp_dir $root_exp_dir/pretrained \
#                  --init_epochs 10 --epochs 0
                 
# python3 train.py --real_image_dir $data_dir/train_photo \
#                  --anime_image_dir $data_dir/Hayao \
#                  --amp --cache --model v2 --num_workers 4 \
#                  --exp_dir $root_exp_dir/pretrained \
#                  --init_epochs 10 --epochs 0


python3 train.py --real_image_dir $data_dir/ffhq-256-jpg-10000 \
                 --anime_image_dir $data_dir/Hayao \
                 --amp --cache --model v2 --num_workers 4 --device cuda:1\
                 --exp_dir $root_exp_dir/pretrained \
                 --init_epochs 10 --epochs 0 --batch_size 16
