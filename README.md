# AnimeGAN Pytorch <a href="https://colab.research.google.com/github/ptran1203/pytorch-animeGAN/blob/master/notebooks/animeGAN_inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a>


Pytorch implementation of AnimeGAN for fast photo animation

* Paper: *AnimeGAN: a novel lightweight GAN for photo animation* - [Semantic scholar](https://www.semanticscholar.org/paper/AnimeGAN%3A-A-Novel-Lightweight-GAN-for-Photo-Chen-Liu/10a9c5d183e7e7df51db8bfa366bc862262b37d7#citing-papers) or from [Yoshino repo](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/doc/Chen2020_Chapter_AnimeGAN.pdf)
* Original implementation in [Tensorflow](https://github.com/TachibanaYoshino/AnimeGAN) by [Tachibana Yoshino](https://github.com/TachibanaYoshino)
* [Try it on Hugging Face](https://huggingface.co/spaces/ptran1203/pytorchAnimeGAN) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ptran1203/pytorchAnimeGAN) 
* [Demo and Docker image on Replicate](https://replicate.ai/ptran1203/pytorch-animegan)
* Sample anime video: https://www.youtube.com/watch?v=45ASFOR3rNU

| Input | Animation |
|--|--|
|![c2](./example/gif/giphy.gif)|![g2](./example/gif/giphy_anime.gif)|
<!-- |![c1](./example/gif/city.gif)|![g1](./example/gif/city_anime.gif)| -->

---
* 09/06/2024: Integrated on Hugging Face Spaces, [try it here](https://huggingface.co/spaces/ptran1203/pytorchAnimeGAN)
* 02/06/2024: Arcane ([result here](#arcane)) and Shinkai style released
* 05/05/2024: Add [color_transfer](https://github.com/ptran1203/color_transfer) module to retain original color of generated images, [See here](#with-color-transfer-module).
* 23/04/2024: Added DDP training.
* 16/04/2024: **AnimeGANv2** (Hayao style) is released with training code
---

## Quick start

```bash
git clone https://github.com/ptran1203/pytorch-animeGAN.git
cd pytorch-animeGAN
```

Run Inference on your local machine
> --src can be directory or image file

```
python3 inference.py --weight hayao:v2 --src /your/path/to/image_dir --out /path/to/output_dir
```

* Python code

```python
from inference import Predictor

predictor= Predictor(
    'hayao:v2',
    # if set True, generated image will retain original color as input image
    retain_color=True
)

url = 'https://github.com/ptran1203/pytorch-animeGAN/blob/master/example/result/real/1%20(20).jpg?raw=true'

predictor.transform_file(url, "anime.jpg")
```

## Pretrained weight

| Model name | Model | Dataset |  Weight |
|--|--|--|--|
| Hayao | AnimeGAN | train_photo + Hayao style | [generator_hayao.pt](https://github.com/ptran1203/pytorch-animeGAN/releases/download/v1.0/generator_hayao.pth) |
| Shinkai | AnimeGAN | train_photo + Shinkai style | [generator_shinkai.pt](https://github.com/ptran1203/pytorch-animeGAN/releases/download/v1.0/generator_shinkai.pth) |
| Hayao:v2 | AnimeGANv2 | Google Landmark v2 + Hayao style | [GeneratorV2_gldv2_Hayao.pt](https://github.com/ptran1203/pytorch-animeGAN/releases/download/v1.2/GeneratorV2_gldv2_Hayao.pt) |
| Shinkai:v2 | AnimeGANv2 | Google Landmark v2 + Shinkai style | [GeneratorV2_gldv2_Shinkai.pt](https://github.com/ptran1203/pytorch-animeGAN/releases/download/v1.2/GeneratorV2_gldv2_Shinkai.pt) |
| Arcane:v2 | AnimeGANv2 | Face ffhq + Arcane style | [GeneratorV2_ffhq_Arcane_210624_e350.pt](https://github.com/ptran1203/pytorch-animeGAN/releases/download/v1.2/GeneratorV2_ffhq_Arcane_210624_e350.pt) |

## Train on custom dataset

- Training notebook on [Google colab](https://colab.research.google.com/github/ptran1203/pytorch-animeGAN/blob/master/notebooks/animeGAN.ipynb)
- Inference notebook on [Google colab](https://colab.research.google.com/github/ptran1203/pytorch-animeGAN/blob/master/notebooks/animeGAN_inference.ipynb)


### 1. Prepare dataset

#### 1.1 To download dataset from the paper, run below command

```bash
wget -O anime-gan.zip https://github.com/ptran1203/pytorch-animeGAN/releases/download/v1.0/dataset_v1.zip
unzip anime-gan.zip
```

=>  The dataset folder can be found in your current folder with named `dataset`

#### 1.2 Create custom data from anime video

You need to have a video file located on your machine.

**Step 1.** Create anime images from the video

```bash
python3 script/video_to_images.py --video-path /path/to/your_video.mp4\
                                --save-path dataset/MyCustomData/style\
                                --image-size 256\
```

**Step 2.** Create edge-smooth version of dataset from **Step 1.**

```bash
python3 script/edge_smooth.py --dataset MyCustomData --image-size 256
```

### 2. Train animeGAN

To train the animeGAN from command line, you can run `train.py` as the following:

```bash
python3 train.py --anime_image_dir dataset/Hayao \
                --real_image_dir dataset/photo_train \
                --model v2 \                 # animeGAN version, can be v1 or v2
                --batch 8 \
                --amp \                      # Turn on Automatic Mixed Precision training
                --init_epochs 10 \
                --exp_dir runs \
                --save-interval 1 \
                --gan-loss lsgan \           # one of [lsgan, hinge, bce]
                --init-lr 1e-4 \
                --lr-g 2e-5 \
                --lr-d 4e-5 \
                --wadvd 300.0\               # Aversarial loss weight for D
                --wadvg 300.0\               # Aversarial loss weight for G
                --wcon 1.5\                  # Content loss weight
                --wgra 3.0\                  # Gram loss weight
                --wcol 30.0\                 # Color loss weight
                --use_sn\                    # If set, use spectral normalization, default is False
```

### 3. Transform images

To convert images in a folder or single image, run `inference.py`, for example:

>
> --src and --out can be a directory or a file

```bash
python3 inference.py --weight path/to/Generator.pt \
                     --src dataset/test/HR_photo \
                     --out inference_images
```

### 4. Transform video

To convert a video to anime version:

> Be careful when choosing --batch-size, it might lead to CUDA memory error if the resolution of the video is too large

```bash
python3 inference.py --weight hayao:v2\
                        --src test_vid_3.mp4\
                        --out test_vid_3_anime.mp4\
                        --batch-size 4
```

#### Result of AnimeGAN v2

##### Hayao

| Input | Hayao style v2 |
|--|--|
|![c1](./example/result/real/1%20(20).jpg)|![g1](./example/result/hayao_v2/1%20(20).jpg)|
|![c1](./example/result/real/1%20(21).jpg)|![g1](./example/result/hayao_v2/1%20(21).jpg)|
|![c1](./example/result/real/1%20(37).jpg)|![g1](./example/result/hayao_v2/1%20(37).jpg)|
|![c1](./example/result/real/1%20(38).jpg)|![g1](./example/result/hayao_v2/1%20(38).jpg)|
|![c1](./example/result/real/1%20(62).jpg)|![g1](./example/result/hayao_v2/1%20(62).jpg)|

##### Arcane

| Input | Arcane |
|--|--|
|![c1](./example/face/leo.jpg)|![g1](./example/arcane/leo.jpg)|
|![c1](./example/face/anne.jpg)|![g1](./example/arcane/anne.jpg)|
|![c1](./example/face/dune2.jpg)|![g1](./example/arcane/dune2.jpg)|
|![c1](./example/face/nat_.jpg)|![g1](./example/arcane/nat_.jpg)|
|![c1](./example/face/seydoux.jpg)|![g1](./example/arcane/seydoux.jpg)|
|![c1](./example/face/tobey.jpg)|![g1](./example/arcane/tobey.jpg)|
|![c1](./example/face/girl4.jpg)|![g1](./example/arcane/girl4.jpg)|


<details>
<summary><strong> More results - Hayao V2 </strong></summary>    

![](./example/more/hayao_v2/pexels-arnie-chou-304906-1004122.jpg)
![](./example/more/hayao_v2/pexels-camilacarneiro-6318793.jpg)
![](./example/more/hayao_v2/pexels-haohd-19859127.jpg)
![](./example/more/hayao_v2/pexels-huy-nguyen-748440234-19838813.jpg)
![](./example/more/hayao_v2/pexels-huy-phan-316220-1422386.jpg)
![](./example/more/hayao_v2/pexels-jimmy-teoh-294331-951531.jpg)
![](./example/more/hayao_v2/pexels-nandhukumar-450441.jpg)
<!-- ![](./example/more/hayao_v2/pexels-sevenstormphotography-575362.jpg) -->
</details>    


<!-- ### Objective:

- Learn to map photo domain **P** to animation domain **A**.
- **AnimeGAN** is Trained using unpaired data includes N photos and M animation images:
    + S(p) = {p(i) | i = 1, ..., N} ⊂ **P**
    + S(a) = {a(i) | i = 1, ..., M} ⊂ **A**
    + S(x) = {x(i) | i = 1, ..., M} ⊂ **X**, grayscale version of **A**
    + S(e) = {e(i) | i = 1, ..., N} ⊂ **E**, Obtained by removing the edges of **A**
    + S(y) = {y(i) | i = 1, ..., N} ⊂ **Y**, grayscale version of **E**

#### Loss functions

- Grayscale Gram matrix to make G(x) have the texture of anime images instread of color (transfer texture, not color)

Loss function

```
L(G, D) = W(adv)L(adv)(G, D) + W(con)L(con)(G, D) + W(gra)L(gra)(G, D) + W(col)L(col)(G,D)
```

1. Adversarial loss (LSGAN)

```
L(adv)(D) = 0.5 * (D(x_anime) - 1)^2 + 0.5 * (D(G(x_photo)))^2

L(adv)(G) = 0.5 (D(G(x_photo)) - 1)^2
```

2. Content loss

```
L(con)(G, D) = ||VGG(x_photo) - VGG(G(x_photo))||
```

3. Gram matrix loss

```
L(gra)(G, D) = ||gram(VGG(G(x_photo))) - Gram(VGG(x_anime_gray))||
```

4. Color recontruction loss

```
L(col)(G, D) = || Y(G(x_photo)) - Y(x_photo) || + Huber(|| U(G(x_photo)) - U(x_photo) ||)
    + Huber(|| V(G(x_photo)) - V(x_photo) ||)
``` -->
