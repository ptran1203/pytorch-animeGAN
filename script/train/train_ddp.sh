#!/bin/bash
set -e
                 
# --wcon  1.5 for Hayao, 2.0 for Paprika, 1.2 for Shinkai
# --wgra 2.5 for Hayao, 0.6 for Paprika, 2.0 for Shinkai
# --wcol 15. for Hayao, 50. for Paprika, 10. for Shinkai
# --wtvar 1. for Hayao, 0.1 for Paprika, 1. for Shinkai
# --init_epochs 100 --resume_G_init /u02/phatth1/runs/v2_gldv2_Hayao_256_50_30/GeneratorV2_images_Hayao_init.pt\

CUDA_VISIBLE_DEVICES=0,1 python3 -u -m torch.distributed.launch --nproc_per_node=2 train.py \
    --real_image_dir /u02/phatth1/dataset/gldv2\
    --anime_image_dir /u02/phatth1/dataset/Hayao\
    --test_image_dir ../dataset/test/HR_photo\
    --amp --model v2 --num_workers 4 --device cuda --ddp\
    --exp_dir /u02/phatth1/runs/v2\
    --init_epochs 0 --resume_G_init /u02/phatth1/runs/v2_gldv2_Hayao_512_25.0/GeneratorV2_gldv2_Hayao_init.pt\
    --debug_samples 0 --epochs 70 --batch_size 4 --use_sn --gan_loss lsgan --d_noise\
    --wadvg 200 --wadvd 200 --wcol 15 --wgra 3.0 --wcon 2.0 --imgsz 352 --wtvar 1.0 --resize_method crop


# Shinkai

CUDA_VISIBLE_DEVICES=0,1 python3 -u -m torch.distributed.launch --nproc_per_node=2 train.py \
    --real_image_dir /u02/phatth1/dataset/gldv2\
    --anime_image_dir /u02/phatth1/dataset/Shinkai\
    --test_image_dir ../dataset/test/HR_photo\
    --amp --model v2 --num_workers 4 --device cuda --ddp\
    --exp_dir /u02/phatth1/runs/v2\
    --resume_G_init ../GeneratorV2_gldv2_Hayao_init.pt\
    --debug_samples 0 --epochs 70 --batch_size 4 --use_sn --gan_loss lsgan\
    --wadvg 30 --wadvd 30 --wcol 25 --wgra 3.0 --wcon 2.0 --imgsz 416 --wtvar 1.0 --resize_method crop

# summerWar
CUDA_VISIBLE_DEVICES=0,1 python3 -u -m torch.distributed.launch --nproc_per_node=2 train.py \
    --real_image_dir /u02/phatth1/dataset/gldv2\
    --anime_image_dir /u02/phatth1/dataset/SummerWar\
    --test_image_dir ../dataset/test/HR_photo\
    --amp --model v2 --num_workers 4 --device cuda --ddp\
    --exp_dir /u02/phatth1/runs/v2\
    --resume_G_init ../GeneratorV2_gldv2_Hayao_init.pt\
    --debug_samples 0 --epochs 70 --batch_size 4 --use_sn --gan_loss lsgan\
    --wadvg 30 --wadvd 30 --wcol 25 --wgra 3.0 --wcon 2.0 --imgsz 416 --wtvar 1.0 --resize_method crop

# Paprika
CUDA_VISIBLE_DEVICES=0,1 python3 -u -m torch.distributed.launch --nproc_per_node=2 train.py \
    --real_image_dir /u02/phatth1/dataset/gldv2\
    --anime_image_dir /u02/phatth1/dataset/Paprika\
    --test_image_dir ../dataset/test/HR_photo\
    --amp --model v2 --num_workers 4 --device cuda --ddp\
    --exp_dir /u02/phatth1/runs/v2\
    --resume_G_init ../GeneratorV2_gldv2_Hayao_init.pt\
    --debug_samples 0 --epochs 70 --batch_size 4 --use_sn --gan_loss lsgan\
    --wadvg 30 --wadvd 30 --wcol 25 --wgra 3.0 --wcon 2.0 --imgsz 416 --wtvar 1.0 --resize_method crop


CUDA_VISIBLE_DEVICES=0,1 python3 -u -m torch.distributed.launch --nproc_per_node=2 train.py \
    --real_image_dir /u02/phatth1/dataset/gldv2\
    --anime_image_dir /u02/phatth1/dataset/Kimetsu\
    --test_image_dir ../dataset/test/HR_photo\
    --amp --model v2 --num_workers 4 --device cuda --ddp\
    --exp_dir /u02/phatth1/runs/v2\
    --resume_G_init ../GeneratorV2_gldv2_Hayao_init.pt\
    --debug_samples 0 --epochs 70 --batch_size 4 --use_sn --gan_loss lsgan\
    --wadvg 30 --wadvd 30 --wcol 25 --wgra 3.0 --wcon 2.0 --imgsz 416 --wtvar 1.0 --resize_method crop
