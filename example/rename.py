import os
from shutil import copyfile


dir_ = './example/result'
readme = './README.md'

def anime_2_input(fi):
    return fi.replace("_anime", "")

def rename(f):
    return f.replace(" ", "").replace("(", "").replace(")", "")


files = os.listdir(dir_)
new_files = []
for f in files:
    input_ver = os.path.join(dir_, anime_2_input(f))

    if not os.path.exists(input_ver):
        # move
        copyfile(f"dataset/test/HR_photo/{anime_2_input(f)}", rename(input_ver))

    os.rename(
        os.path.join(dir_, f),
        os.path.join(dir_, rename(f))
    )
