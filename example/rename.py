import os

dir_ = './example/result'
files = os.listdir(dir_)

for f in files:
    new_name = f.replace(" ", "").replace("(", "").replace(")", "")
    os.rename(
        os.path.join(dir_, f),
        os.path.join(dir_, new_name)
    )