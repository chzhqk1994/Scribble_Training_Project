import os
import cv2
import json
import shutil


def make_dir_list(src_dir, separated_dir):
    src_img_dir_list = []
    dst_img_dir_list = []
    src_anno_dir_list = []
    dst_anno_dir_list = []
    src_anno_dir_list.append(os.path.join(src_dir, 'json'))
    dst_anno_dir_list.append(os.path.join(separated_dir, 'json'))
    src_img_dir_list.append(os.path.join(src_dir, 'original_png'))
    dst_img_dir_list.append(os.path.join(separated_dir, 'original_png'))

    return src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list


def is_exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def json_parser(file_path, label_list):
    with open(file_path) as json_file:
        json_data = json.load(json_file)
        shapes = json_data["shapes"]
        new_shapes = []
        height = int(json_data["imageHeight"])
        width = json_data["imageWidth"]
        have_class = False
        for shape in shapes:
            if not shape["label"] in label_list:  # find 대상이 아니라면 pass
                continue
            else:  # find 대상이면  return true
                have_class = True

        return have_class

def walk_around_files(label_list, src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list, JUST_FIND):
    err_file_cnt = 0
    for src_dir in src_anno_dir_list:
        index = src_anno_dir_list.index(src_dir)
        err_file_cnt = 0
        if not JUST_FIND:
            is_exist_dir(dst_img_dir_list[index])
            is_exist_dir(dst_anno_dir_list[index])
        for root, dirs, files in os.walk(src_dir):
            print('srcdir:', src_dir, " index : ", index)
            print('dstdir:', dst_anno_dir_list[index])
            for file in files:
                img_ext = 'png'
                # current_anno_file = os.path.join(str(root), file)
                # current_img_file = os.path.join(src_img_dir_list[index], file[:-4] + img_ext)

                current_anno_file = os.path.join(src_dir, file)
                current_img_file = os.path.join(src_img_dir_list[index], file[:-4] + img_ext)

                if not os.path.isfile(current_img_file):
                    img_ext = 'PNG'
                    current_img_file = os.path.join(src_img_dir_list[index], file[:-4] + img_ext)

                dst_anno_file = os.path.join(dst_anno_dir_list[index], file)
                dst_img_file = os.path.join(dst_img_dir_list[index], file[:-4] + img_ext)

                have_class = json_parser(current_anno_file, label_list)
                if have_class:
                    print('Finding class here', current_anno_file)
                    if not JUST_FIND:
                        try:
                            shutil.copy2(current_img_file, dst_img_file)
                        except FileNotFoundError:
                            print('[IMG ERROR] ',file)

                        try:
                            shutil.copy2(current_anno_file, dst_anno_file)
                        except FileNotFoundError:
                            print('[ANNO ERROR] ',file)

    print("Error : ", err_file_cnt)


if __name__ == "__main__":
    label_list = ['parking_tower']  # find class
    src_dir = '/media/qisens/4tb3/kowa_global/parkinglot_detection/dataset/scribble/parking_tower/'
    separated_dir = '/media/qisens/4tb3/kowa_global/parkinglot_detection/dataset/scribble/parking_tower/parking_tower_images/'
    JUST_FIND = False  # just print if json have specific class, if u want separate, set to False

    src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list = make_dir_list(src_dir, separated_dir)
    walk_around_files(label_list, src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list, JUST_FIND)
