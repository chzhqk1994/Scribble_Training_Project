import json
import os
import cv2
import matplotlib as mpl
import numpy as np
mpl.use('Agg')

def is_exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def make_dir_list(theta_ls, src_dir, separated_dir):

    dst_img_dir_list = []
    dst_anno_dir_list = []
    src_anno_dir_list = os.path.join(src_dir,'json')
    src_img_dir_list = os.path.join(src_dir, 'original_png')
    for theta in theta_ls:
        dst_anno_dir_list.append(os.path.join(separated_dir + str(theta), 'json'))
        dst_img_dir_list.append(os.path.join(separated_dir + str(theta), 'original_png'))

    return src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list

def rotate_point(point,width,height, theta):
    rad = np.deg2rad(theta)
    rotate_matrix = [[np.cos(rad), np.sin(rad)],[-np.sin(rad),np.cos(rad)]]
    x = point[0] - width/2
    y = point[1] - height/2
    for i in range(len(point)):
        point[i] = x*rotate_matrix[i][0] + y*rotate_matrix[i][1]

    point[0] = point[0] + width/2
    point[1] = point[1] + height/2
    return point

def json_parser(file_path, key, theta):
    with open(file_path) as json_file:
        json_data = json.load(json_file)
        shapes = json_data["shapes"]
        new_shapes = []
        height = int(json_data["imageHeight"])
        width = json_data["imageWidth"]
        for shape in shapes:
            if shape["label"] in key :
                label_points = {}
                label_points["label"] = shape["label"]
                label_points["points"] = []
                for point in shape["points"]:
                    rotated_point = rotate_point(point,width,height,theta)
                    label_points["points"].append(rotated_point)
                label_points["group_id"] = shape["group_id"]
                label_points["shape_type"] = shape["shape_type"]
                label_points["flags"] = shape["flags"]

                new_shapes.append(label_points)
            else :
                continue

        if new_shapes:
            new_json = {}
            new_json["shapes"] = new_shapes
            new_json["imagePath"] = json_data["imagePath"][:-4] + "_" + str(theta) + ".png"
            new_json["imageData"] = json_data["imageData"]
            new_json["imageHeight"] = json_data["imageHeight"]
            new_json["imageWidth"] = json_data["imageWidth"]
            return new_json
        else:
            return {}

def rotate_images(theta_ls, src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list):
    err_file_cnt = 0
    for theta in theta_ls:
        index = theta_ls.index(theta)
        is_exist_dir(dst_img_dir_list[index])
        is_exist_dir(dst_anno_dir_list[index])
        for root, dirs, files in os.walk(src_anno_dir_list):
            print('srcdir:', src_anno_dir_list, " index : ", index)
            for file in files:
                current_anno_file = os.path.join(str(root), file)
                current_img_file = os.path.join(src_img_dir_list, file[:-4] + 'png')
                if not os.path.isfile(current_img_file):
                    current_img_file = os.path.join(src_img_dir_list, file[:-4] + 'PNG')

                rotate_tag = '_' + str(theta)
                dst_anno_file = os.path.join(dst_anno_dir_list[index], file[:-5] + rotate_tag + '.json')
                dst_img_file = os.path.join(dst_img_dir_list[index], file[:-5] + rotate_tag + '.png')

                key = ['bg', 'waterproof', 'facility', 'something', 'airconditioner', 'wall', "garden", "corrugated", "slate", "solarpanel", "heliport", "tree", "concrete", "clay"]

                new_json = json_parser(current_anno_file, key, theta)
                if new_json:
                    # To Do
                    try:

                        print(file)
                        print(current_img_file)
                        current_img = cv2.imread(current_img_file)
                        origin_height, origin_width, _ = current_img.shape
                        (cx_origin, cy_origin) = (origin_width // 2, origin_height // 2)
                        M = cv2.getRotationMatrix2D((cx_origin, cy_origin), theta, 1.0)
                        rotated_img = cv2.warpAffine(current_img, M, (origin_width, origin_height))
                        #new_height, new_width = rotated_img.shape
                        cv2.imwrite(dst_img_file, rotated_img)

                        with open(dst_anno_file, 'w', encoding="utf-8") as anno_file:
                            json.dump(new_json, anno_file, ensure_ascii=False, indent="\t")
                    except FileNotFoundError as e:
                        print('ERROR : ', file)
                        err_file_cnt += 1
                        continue
                else:
                    continue
                # To DO
    print("Error : ", err_file_cnt)

if __name__ == "__main__":
    theta_ls = [-30,-60, 30, 60]
    #src_dir_folder = ['original_png', 'scribble_png']
    src_dir = "/media/qisens/4tb3/navermap_detection/dataset/scribble_dataset/building_capture/augmentation/origin/"
    separated_dir = "/media/qisens/4tb3/navermap_detection/dataset/scribble_dataset/building_capture/augmentation/rotated_"



    src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list = make_dir_list(theta_ls, src_dir, separated_dir)
    rotate_images(theta_ls, src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list)
