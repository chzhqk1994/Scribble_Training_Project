import json
import os
import cv2
import matplotlib as mpl
import numpy as np
mpl.use('Agg')

def is_exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def change_image_path_on_json(src_dir):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path) as json_file:
                json_data = json.load(json_file)
                json_data["imagePath"] = "../JPEGImages/" + json_data["imagePath"].split('/')[-1]

            with open(file_path, 'w', encoding="utf-8") as result_file:
                json.dump(json_data, result_file, ensure_ascii=False, indent="\t")


if __name__ == "__main__":
    src_dir = "/media/qisens/2tb1/python_projects/training_pr/Scribble/input/seoulgrinizing/JSONScribble"

    change_image_path_on_json(src_dir)