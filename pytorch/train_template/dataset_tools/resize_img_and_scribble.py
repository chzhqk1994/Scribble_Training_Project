import json
import os
import cv2
import numpy as np

def is_exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def json_parser(file_path, resize_ratio):
    with open(file_path) as json_file:
        json_data = json.load(json_file)
        shapes = json_data["shapes"]
        new_shapes = []
        height = int(json_data["imageHeight"])
        width = json_data["imageWidth"]
        for shape in shapes:

            label_points = {}
            label_points["label"] = shape["label"]
            label_points["points"] = []

            for point in shape["points"]:
                new_point = []
                new_point_x = (point[0] * resize_ratio) + (1 - resize_ratio)*width/2
                new_point_y = (point[1] * resize_ratio) + (1 - resize_ratio)*height/2
                new_point.append(new_point_x)
                new_point.append(new_point_y)
                label_points["points"].append(new_point)
            label_points["group_id"] = shape["group_id"]
            label_points["shape_type"] = shape["shape_type"]
            label_points["flags"] = shape["flags"]

            new_shapes.append(label_points)

        if new_shapes:
            new_json = {}
            new_json["shapes"] = new_shapes
            new_json["imagePath"] = json_data["imagePath"][:-4] + "_" + str(resize_ratio) + ".png"
            new_json["imageData"] = json_data["imageData"]
            new_json["imageHeight"] = json_data["imageHeight"]
            new_json["imageWidth"] = json_data["imageWidth"]
            return new_json
        else :
            return {}


def make_dir_list(src_dir, separated_dir):

    src_img_dir_list = []
    dst_img_dir_list = []
    src_anno_dir_list = []
    dst_anno_dir_list = []
    #src_anno_dir_list.append(os.path.join(src_dir,'not_aug_json'))
    src_anno_dir_list.append(os.path.join(src_dir, 'json'))
    dst_anno_dir_list.append(os.path.join(separated_dir, 'json'))
    src_img_dir_list.append(os.path.join(src_dir, 'original_png'))
    dst_img_dir_list.append(os.path.join(separated_dir, 'original_png'))

    return src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list

def make_noise(resized_image, resized_width, resized_height, origin_height, origin_width):
    # noise_img = (np.uint8(np.random.rand(origin_width, origin_height, 3) * 255))
    noise_img = np.zeros([origin_height, origin_width, 3])

    horizontal_padding = int((origin_width - resized_width) / 2)
    vertical_padding = int((origin_height - resized_height) / 2)
    noise_img[vertical_padding:vertical_padding + resized_height,
              horizontal_padding:horizontal_padding + resized_width] = resized_image
    return noise_img

def cut_image(src_image, new_height, new_width, origin_height, origin_width):
    horizontal_extra = int((new_width - origin_width) / 2)
    vertical_extra = int((new_height - origin_height) / 2)
    cutted_image = src_image[horizontal_extra:horizontal_extra + origin_height,
                             vertical_extra: vertical_extra + origin_width]

    return cutted_image

def resize_image(src_image, new_height, new_width, origin_height, origin_width, resize_ratio):
    resized_image = cv2.resize(src_image, (new_width, new_height))

    # resize 비율이 1보다 크면 확대이므로 가우시안 노이즈가 필요없고, 원본사이즈 유지이므로 사이즈범위 밖의 테두리 부분은 잘라냄
    if resize_ratio > 1:
        cutted_image = cut_image(resized_image, new_height, new_width, origin_height, origin_width)
        return cutted_image

    # resize 비율이 1보다 작으면 축소이므로 원본사이즈보다 줄어든만큼 가우시안 노이즈가 필요함
    else:
        noised_image = make_noise(resized_image, new_width, new_height, origin_height, origin_width)
        return noised_image

def walk_around_files(resize_ratio, src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list):
    err_file_cnt = 0
    for src_dir in src_anno_dir_list:
        index = src_anno_dir_list.index(src_dir)
        is_exist_dir(dst_img_dir_list[index])
        is_exist_dir(dst_anno_dir_list[index])
        for root, dirs, files in os.walk(src_dir):
            print('srcdir:', src_dir, " index : ", index)
            for file in files:
                #current_scrib_file = os.path.join(src_scrib_dir_list[index], file[:-4] + 'png')
                current_anno_file = os.path.join(str(root),file)
                current_img_file = os.path.join(src_img_dir_list[index], file[:-4] + 'png')
                if not os.path.isfile(current_img_file):
                    current_img_file = os.path.join(src_img_dir_list[index], file[:-4] + 'PNG')

                resize_tag = '_' + str(resize_ratio)
                #dst_scrib_file = os.path.join(dst_scrib_dir_list[index], file[:-5] + resize_tag + '.png')
                dst_anno_file = os.path.join(dst_anno_dir_list[index], file[:-5] + resize_tag + '.json')
                dst_img_file = os.path.join(dst_img_dir_list[index], file[:-5] + resize_tag + '.png')

                new_json = json_parser(current_anno_file, resize_ratio)
                if new_json :
                # To Do
                    try:

                        print(file)
                        print(current_img_file)
                        current_img = cv2.imread(current_img_file)
                        origin_height, origin_width, _ = current_img.shape  #640,640
                        new_height, new_width = int(origin_height * resize_ratio), int(origin_width * resize_ratio)
                        resized_img = resize_image(current_img, new_height, new_width, origin_height, origin_width, resize_ratio)  # 이미지 resize
                        print(origin_height, origin_width)
                        cv2.imwrite(dst_img_file, resized_img)

                        with open(dst_anno_file,'w',encoding="utf-8") as anno_file :
                            json.dump(new_json, anno_file, ensure_ascii=False, indent="\t")
                    except FileNotFoundError as e:
                        print('ERROR : ', file)
                        err_file_cnt += 1
                        continue
                else :
                    continue
                # To DO
    print("Error : ", err_file_cnt)

if __name__ == "__main__":
    resize_ratio_list = [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4]
    #src_dir_folder = ['original_png', 'scribble_png']
    src_dir = "/media/qisens/4tb3/kowa_global/parkinglot_detection/dataset/scribble/parking_tower/parking_tower_data/merged/origin/"
    dst_dir = "/media/qisens/4tb3/kowa_global/parkinglot_detection/dataset/scribble/parking_tower/parking_tower_data/merged/resize_"

    for resize_ratio in resize_ratio_list:
        separated_dir = dst_dir + str(resize_ratio)

        # resize 를 하더라도 원본 크기는 유지하여 layer 에 입력되는 object 의 크기를 다르게 함
        # 640x640 보다 크게 확장되더라도 나머지부분은 자름
        # 640x640 보다 작게 축소되더라도 나머지부분은 가우시안 노이즈로 채움

        src_anno_dir_list,dst_anno_dir_list, src_img_dir_list, dst_img_dir_list = make_dir_list(src_dir,
                                                                                                 separated_dir)
        walk_around_files(resize_ratio, src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list)
