import os
from random import shuffle

# -----------------------------------------------------------------------------if val / train dataset are not splited
dir_path = "/media/qisens/4tb3/kowa_global/parkinglot_detection/dataset/scribble/googlemap_added/original_png/"
label_path = "/media/qisens/4tb3/kowa_global/parkinglot_detection/dataset/scribble/googlemap_added/json/"
path, dirs, files = next(os.walk(dir_path))
label_path, label_dirs, label_files = next(os.walk(label_path))

#shuffle(files)

# strain/val --  8:2
file_cnt = len(files)
train_cnt = round(file_cnt * 0.9)

train_files = files[:train_cnt]
shuffle(train_files)

val_files = files[train_cnt:]

train_txt = "/media/qisens/4tb3/kowa_global/parkinglot_detection/dataset/scribble/googlemap_added/split_data/train.txt"
val_txt = "/media/qisens/4tb3/kowa_global/parkinglot_detection/dataset/scribble/googlemap_added/split_data/val.txt"

with open(train_txt, "w") as file1:
    for idx, f in enumerate(train_files):
        name, ext = os.path.splitext(f)
        if not name + '.json' in label_files:
            print(f'{f} No has label')
            continue
        file1.write(name + "\n")
file1.close()
with open(val_txt, "w") as file2:
    for f in val_files:
        name, ext = os.path.splitext(f)
        if not name + '.json' in label_files:
            print(f'{f} No has label')
            continue
        file2.write(name + "\n")
file2.close()

