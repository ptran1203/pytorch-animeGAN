# AnimeGAN Pytorch

Pytorch implementation of AnimeGAN for fast photo animation

* Paper: *AnimeGAN: a novel lightweight GAN for photo animation* - [Semantic scholar](https://www.semanticscholar.org/paper/AnimeGAN%3A-A-Novel-Lightweight-GAN-for-Photo-Chen-Liu/10a9c5d183e7e7df51db8bfa366bc862262b37d7#citing-papers) or from [Yoshino repo](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/doc/Chen2020_Chapter_AnimeGAN.pdf)
* Original implementation in [Tensorflow](https://github.com/TachibanaYoshino/AnimeGAN) by [Tachibana Yoshino](https://github.com/TachibanaYoshino)


| Input | Animation |
|--|--|
|![c1](./example/video_city_small.gif)|![g1](./example/video_city_anime.gif)|

## Documentation

### 1. Train animeGAN

To train the animeGAN from command line, you can run `train.py` as the following:

```bash
python3 train.py --batch 6\
                --init-epochs 4\
                --checkpoint-dir {ckp_dir}\
                --save-image-dir {save_img_dir}\
                --save-interval 1\
                --gan-loss lsgan\           # one of [lsgan, hinge, bce]
                --init-lr 0.0001\
                --lr-g 0.00002\
                --lr-d 0.00004\
                --wadvd 10.0\               # Aversarial loss weight for D
                --wadvg 10.0\               # Aversarial loss weight for G
                --wcon 1.5\                 # Content loss weight
                --wgra 3.0\                 # Gram loss weight
                --wcol 30.0\                # Color loss weight
                --continu GD\               # if set, G to start from pre-trained G, GD to continue training GAN
                --use_sn\                   # If set, use spectral normalization, default is False
```

### 2. Transform images

To convert images in a folder or single image, run `inference_image.py`, for example:

> --src and --dest can be a directory or a file

```bash
python3 inference_image.py --checkpoint {ckp_dir}\
                        --src /content/test/HR_photo\
                        --dest {working_dir}/inference_image_v2\
```

### 3. Transform video

To convert a video to anime version, run `inference_video.py`, for example:

> Be careful when choosing --batch-size, it might lead to CUDA memory error if the resolution of the video is too large

```bash
python3 inference_video.py --checkpoint {ckp_dir}\
                        --src /content/test_vid_3.mp4\
                        --dest /content/test_vid_3_anime.mp4\
                        --batch-size 2
```

## Anime transformation results


| Input | Output |
|--|--|
|![c1](./example/result/136.jpg)|![g1](./example/result/136_anime.jpg)|
|![c1](./example/result/137.jpg)|![g1](./example/result/137_anime.jpg)|
|![c1](./example/result/139.jpg)|![g1](./example/result/139_anime.jpg)|
|![c1](./example/result/142.jpg)|![g1](./example/result/142_anime.jpg)|
|![c1](./example/result/146.jpg)|![g1](./example/result/146_anime.jpg)|
|![c1](./example/result/148.jpg)|![g1](./example/result/148_anime.jpg)|


## Check list

- [ ] Add Google Colab
- [ ] Add implementation details
- [ ] Add and train on other data
- [ ] ...

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
