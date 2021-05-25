import os
import cv2
import re

REG = re.compile(r"[0-9]{3}")
dir_ = './example/result'
readme = './README.md'


def anime_2_input(fi):
    return fi.replace("_anime", "")

def rename(f):
    return f.replace(" ", "").replace("(", "").replace(")", "")

def rename_back(f):
    nums = REG.search(f)
    if nums:
        nums = nums.group()
        return f.replace(nums, f"{nums[0]} ({nums[1:]})")

    return f.replace('jpeg', 'jpg')

def copyfile(src, dest):
    # copy and resize
    im = cv2.imread(src)

    if im is None:
        raise FileNotFoundError(src)

    h, w = im.shape[1], im.shape[0]

    s = 448
    size = (s, round(s * w / h))
    im = cv2.resize(im, size)

    print(w, h, im.shape)
    cv2.imwrite(dest, im)

files = os.listdir(dir_)
new_files = []
for f in files:
    input_ver = os.path.join(dir_, anime_2_input(f))
    copyfile(f"dataset/test/HR_photo/{rename_back(anime_2_input(f))}", rename(input_ver))

    os.rename(
        os.path.join(dir_, f),
        os.path.join(dir_, rename(f))
    )
