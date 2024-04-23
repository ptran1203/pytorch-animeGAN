
# V1

data_dir=/u02/phatth1/dataset
root_exp_dir=/u02/phatth1/runs


# # Ver1: Hayao
# python3 train.py --real_image_dir $data_dir/landscape \
#                  --anime_image_dir $data_dir/Hayao \
#                  --amp --model v1 --num_workers 4 --device cuda:0\
#                  --exp_dir $root_exp_dir/v1_landscape_Hayao \
#                  --resume_G_init /u02/phatth1/runs/pretrained/GeneratorV1_landscape_10e.pt

# Ver2: Hayao
python3 train.py --real_image_dir $data_dir/landscape \
                 --anime_image_dir $data_dir/Hayao \
                 --amp --model v2 --num_workers 4 --device cuda:1\
                 --exp_dir $root_exp_dir/v2_landscape_Hayao \
                 --resume_G_init /u02/phatth1/runs/pretrained/GeneratorV2_landscape_10e.pt \
                 --test_image_dir ../dataset/test/HR_photo --debug_samples 0

# Ver2: 
# python3 train.py --real_image_dir $data_dir/ffhq \
#                  --anime_image_dir $data_dir/Hayao \
#                  --amp --model v2 --num_workers 4 --device cuda:0\
#                  --exp_dir $root_exp_dir/v2_ffhq_Hayao \
#                  --resume_G_init /u02/phatth1/runs/pretrained/GeneratorV2_ffhq_10e.pt
