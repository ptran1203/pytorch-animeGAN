py train.py --real_image_dir dataset/train_photo^
            --anime_image_dir dataset/Hayao^
            --test_image_dir dataset/test/HR_photo^
            --amp --model v2 --num_workers 4 --exp_dir runs/v2^
            --init_epochs 10 --epochs 100 --batch_size 4 --use_sn^
            --gan_loss lsgan --wadvg 200 --wadvd 200 --wcol 15 --wgra 3.0^
            --wcon 2.0 --imgsz 256 --wtvar 1.0 --resize_method resize --resume

@REM --resume_G_init runs/v2_train_photo_Hayao_256_200.0/GeneratorV2_train_photo_Hayao_init.pt