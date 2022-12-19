from PIL import Image, ImageDraw, ImageOps
from dataloaders import parameters_tree
from collections import Counter
import json
import os

label_dict = parameters_tree.LABEL_DICT

# you should give full path
dir_path = "/media/qisens/4tb3/kowa_global/parkinglot_detection/dataset/scribble/parking_tower/augmented_parkingtower+origindataset/json"
scribble_path = "/media/qisens/4tb3/kowa_global/parkinglot_detection/dataset/scribble/parking_tower/augmented_parkingtower+origindataset/scribble_png"
if not os.path.exists(scribble_path):
    os.makedirs(scribble_path)

label_cnt = []
for path, dirs, files in os.walk(dir_path):
    for file in files:
        file_path = os.path.join(path, file)
        print(file_path)

        try:
            f = open(file_path)
            data = json.load(f)
        except json.decoder.JSONDecodeError:
            print("ERROR: ", file_path)

        #image_path = os.path.join(scribble_path, data["imagePath"])
        image_path = os.path.join(scribble_path, file[:-4]) + 'png'
        width, height = data["imageWidth"], data["imageHeight"]
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        for line in data["shapes"]:
            label = line["label"]
            points = line["points"]

            label_cnt.append(label)

            for idx, point in enumerate(points):
                if idx == 0:
                    first_pt = point
                else:
                    second_pt = point
                    draw.line((first_pt[0], first_pt[1], second_pt[0], second_pt[1]),
                              fill=(label_dict[label], label_dict[label], label_dict[label]), width=5)
                    first_pt = second_pt

        gray_image = ImageOps.grayscale(img)
        gray_image.save(image_path)

print(Counter(label_cnt))
