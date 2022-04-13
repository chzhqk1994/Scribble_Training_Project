import os
from random import shuffle

# -----------------------------------------------------------------------------if val / train dataset are not splited
dir_path = "/media/qisens/4tb3/navermap_detection/dataset/scribble_dataset/building_capture/flat_facility_round_edge_corrugated_slate_panel_garden_heliport_window_nowall_added_added/original_png/"
path, dirs, files = next(os.walk(dir_path))
#shuffle(files)

# strain/val --  8:2
file_cnt = len(files)
train_cnt = round(file_cnt * 0.8)

train_files = files[:train_cnt]
shuffle(train_files)

val_files = files[train_cnt:]

train_txt = "/media/qisens/4tb3/navermap_detection/dataset/scribble_dataset/building_capture/flat_facility_round_edge_corrugated_slate_panel_garden_heliport_window_nowall_added_added/split_data/train.txt"
val_txt = "/media/qisens/4tb3/navermap_detection/dataset/scribble_dataset/building_capture/flat_facility_round_edge_corrugated_slate_panel_garden_heliport_window_nowall_added_added/split_data/val.txt"

with open(train_txt, "w") as file1:
    for idx, f in enumerate(train_files):
        name, ext = os.path.splitext(f)
        file1.write(name + "\n")
file1.close()
with open(val_txt, "w") as file2:
    for f in val_files:
        name, ext = os.path.splitext(f)
        file2.write(name + "\n")
file2.close()
