import os
import json


def make_both_exists_file_list(src_dir, dst_dir):
    exist_file = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if os.path.exists(os.path.join(dst_dir, file)):
                exist_file.append(file)

    return exist_file


def copy_classes_to_dst_dataset(src_dir, dst_dir, target_classes):
    exist_file = make_both_exists_file_list(src_dir, dst_dir)
    for file in exist_file:
        src_json = os.path.join(src_dir, file)
        dst_json = os.path.join(dst_dir, file)

        src_json_file = open(src_json, 'r')
        src_json_obj = json.load(src_json_file)
        src_json_file.close()
        src_class_list = src_json_obj['shapes']
        extract_class_list = []

        # src 데이터셋에서 타겟 레이블 정보들을 추출
        for class_dict in src_class_list:
            if class_dict['label'] in target_classes:
                extract_class_list.append(class_dict)

        # src 데이터셋에서 추출된 레이블 정보들을 dst 데이터셋에 추가
        dst_json_file = open(dst_json, 'r')
        dst_json_obj = json.load(dst_json_file)
        dst_json_obj['shapes'] += extract_class_list
        dst_json_file.close()

        with open(dst_json, 'w') as dst_json_file:
            json.dump(dst_json_obj, dst_json_file, indent=2)


if __name__ == "__main__":
    # src_dir 데이터셋의 특정 클래스들을 dst_dir 의 데이터셋에 copy
    target_classes = ["facility"]
    src_dir = './from/json/'
    dst_dir = './to/json/'

    copy_classes_to_dst_dataset(src_dir, dst_dir, target_classes)
