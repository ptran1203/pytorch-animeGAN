import os

dir_ = './example/result'
readme = './README.md'

files = os.listdir(dir_)
new_files = []
for f in files:
    new_name = f.replace(" ", "").replace("(", "").replace(")", "")
    os.rename(
        os.path.join(dir_, f),
        os.path.join(dir_, new_name)
    )
