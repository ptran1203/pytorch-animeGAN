#!/bin/bash

                 
# --wcon  1.5 for Hayao, 2.0 for Paprika, 1.2 for Shinkai
# --wgra 2.5 for Hayao, 0.6 for Paprika, 2.0 for Shinkai
# --wcol 15. for Hayao, 50. for Paprika, 10. for Shinkai
# --wtvar 1. for Hayao, 0.1 for Paprika, 1. for Shinkai


 CUDA_VISIBLE_DEVICES=0,1 python3 -u -m torch.distributed.launch --nproc_per_node=2 train.py \
    --real_image_dir /u02/phatth1/dataset/gldv2_micro/images\
    --anime_image_dir /u02/phatth1/dataset/Hayao\
    --test_image_dir ../dataset/test/HR_photo\
    --amp --model v2 --num_workers 4 --device cuda --ddp\
    --exp_dir /u02/phatth1/runs/v2_gldv2_Hayao\
    --debug_samples 0 --epochs 100 --init_epochs 0 --batch_size 4\
    --resume_G_init /u02/phatth1/runs/v2_gldv2_Hayao/GeneratorV2_images_Hayao_init.pt\
    --wcol 100 --wgra 5.0 --wcon 1.0 --imgsz 256 --wtvar 1.0

# 5 mins
sleep 300


CUDA_VISIBLE_DEVICES=0,1 python3 -u -m torch.distributed.launch --nproc_per_node=2 train.py \
    --real_image_dir /u02/phatth1/dataset/gldv2_micro/images\
    --anime_image_dir /u02/phatth1/dataset/Shinkai\
    --test_image_dir ../dataset/test/HR_photo\
    --amp --model v2 --num_workers 4 --device cuda --ddp\
    --exp_dir /u02/phatth1/runs/v2_gldv2_Shinkai\
    --debug_samples 0 --epochs 100 --init_epochs 0 --batch_size 4\
    --resume_G_init /u02/phatth1/runs/v2_gldv2_Hayao/GeneratorV2_images_Hayao_init.pt\
    --wcol 100 --wgra 5.0 --wcon 1.0 --imgsz 256 --wtvar 1.0

sleep 300

CUDA_VISIBLE_DEVICES=0,1 python3 -u -m torch.distributed.launch --nproc_per_node=2 train.py \
    --real_image_dir /u02/phatth1/dataset/ffhq\
    --anime_image_dir /u02/phatth1/dataset/Hayao\
    --test_image_dir ../dataset/test/HR_photo\
    --amp --model v2 --num_workers 4 --device cuda --ddp\
    --exp_dir /u02/phatth1/runs/v2_ffhq_Hayao\
    --debug_samples 0 --epochs 100 --init_epochs 10 --batch_size 4\
    --wcol 100 --wgra 5.0 --wcon 1.0 --imgsz 256 --wtvar 1.0

# CUDA_VISIBLE_DEVICES=0,1 python3 -u -m torch.distributed.launch --nproc_per_node=2 train.py \
#     --real_image_dir /u02/phatth1/dataset/gldv2_micro/images\
#     --anime_image_dir /u02/phatth1/dataset/Hayao\
#     --test_image_dir ../dataset/test/HR_photo\
#     --amp --model v2 --num_workers 4 --device cuda --ddp\
#     --exp_dir /u02/phatth1/runs/v2_gldv2_Hayao512\
#     --debug_samples 0 --epochs 70 --init_epochs 10 --batch_size 4\
#     --wcol 10 --wgra 2.0 --wcon 1.2 --imgsz 512

# # 5 mins
sleep 300

CUDA_VISIBLE_DEVICES=0,1 python3 -u -m torch.distributed.launch --nproc_per_node=2 train.py \
    --real_image_dir /u02/phatth1/dataset/gldv2_micro/images\
    --anime_image_dir /u02/phatth1/dataset/Paprika\
    --test_image_dir ../dataset/test/HR_photo\
    --amp --model v2 --num_workers 4 --device cuda --ddp\
    --exp_dir /u02/phatth1/runs/v2_gldv2_Paprika\
    --debug_samples 0 --epochs 100 --init_epochs 0 --batch_size 4\
    --resume_G_init /u02/phatth1/runs/v2_gldv2_Hayao/GeneratorV2_images_Hayao_init.pt\
    --wcol 100 --wgra 5.0 --wcon 1.0 --imgsz 256 --wtvar 1.0