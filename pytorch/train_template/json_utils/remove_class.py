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
        for shape in shapes:
            if not shape["label"] in label_list:  # 삭제대상이 아니라면 list 에 append
                label_points = {}
                label_points["label"] = shape["label"]
                label_points["points"] = shape['points']
                label_points["group_id"] = shape["group_id"]
                label_points["shape_type"] = shape["shape_type"]
                label_points["flags"] = shape["flags"]
                new_shapes.append(label_points)

            else:  # 삭제대상이면 제거
                continue

        if new_shapes:
            new_json = {}
            new_json["shapes"] = new_shapes
            new_json["imagePath"] = json_data["imagePath"]
            new_json["imageData"] = json_data["imageData"]
            new_json["imageHeight"] = json_data["imageHeight"]
            new_json["imageWidth"] = json_data["imageWidth"]
            return new_json
        else:
            return {}


def walk_around_files(label_list, src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list):
    err_file_cnt = 0
    for src_dir in src_anno_dir_list:
        index = src_anno_dir_list.index(src_dir)
        err_file_cnt = 0
        is_exist_dir(dst_img_dir_list[index])
        is_exist_dir(dst_anno_dir_list[index])
        for root, dirs, files in os.walk(src_dir):
            print('srcdir:', src_dir, " index : ", index)
            print('dstdir:', dst_anno_dir_list[index])
            for file in files:
                img_ext = 'png'
                current_anno_file = os.path.join(str(root), file)
                current_img_file = os.path.join(src_img_dir_list[index], file[:-4] + img_ext)

                if not os.path.isfile(current_img_file):
                    img_ext = 'PNG'
                    current_img_file = os.path.join(src_img_dir_list[index], file[:-4] + img_ext)

                dst_anno_file = os.path.join(dst_anno_dir_list[index], file)
                dst_img_file = os.path.join(dst_img_dir_list[index], file[:-4] + img_ext)

                print(current_img_file)
                print(current_anno_file)
                new_json = json_parser(current_anno_file, label_list)
                shutil.copy2(current_img_file, dst_img_file)
                if new_json:
                    pass
                    with open(dst_anno_file, 'w', encoding="utf-8") as anno_file:
                        json.dump(new_json, anno_file, ensure_ascii=False, indent="\t")
                else:
                    pass
    print("Error : ", err_file_cnt)


if __name__ == "__main__":
    label_list = ['wall']  # will be removed
    src_dir = '/home/qisens/dataset/roof_segment_augment/all/'
    separated_dir = '/home/qisens/dataset/roof_segment_augment/removed/'

    src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list = make_dir_list(src_dir, separated_dir)
    walk_around_files(label_list, src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list)
