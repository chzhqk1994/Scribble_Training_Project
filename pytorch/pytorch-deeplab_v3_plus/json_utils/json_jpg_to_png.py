import os
import io
import cv2
import json
import base64
import numpy as np
from PIL import Image


def json_imageData_to_cv2_image(imageData):
    img_data = base64.b64decode(imageData)
    f = io.BytesIO()
    f.write(img_data)
    img_pil = Image.open(f)
    cv2_image = np.array(img_pil)
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return cv2_image


def json_imageData_to_pil_image(imageData):
    img_data = base64.b64decode(imageData)
    f = io.BytesIO()
    f.write(img_data)
    img_pil = Image.open(f)
    return img_pil


def pil_img_to_b64(img_pil, format):
    f = io.BytesIO()
    img_pil.save(f, format="PNG")
    img_bin = f.getvalue()

    if format == 'byte':
        if hasattr(base64, "encodebytes"):
            img_b64 = base64.encodebytes(img_bin)
        else:
            img_b64 = base64.encodestring(img_bin)
        return img_b64

    elif format == 'str':
        img_b64_str = base64.b64encode(img_bin).decode("utf-8")
        return img_b64_str


def jpg_to_png(src_dir, dst_dir):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension != '.json':
                continue

            src_json = os.path.join(src_dir, file)
            dst_json = os.path.join(dst_dir, file)
            dst_image = os.path.join(dst_dir, filename + '.PNG')

            src_json_file = open(src_json, 'r')
            src_json_obj = json.load(src_json_file)
            src_json_file.close()
            src_jpg_image = src_json_obj['imageData']
            # cv2_image = json_imageData_to_cv2_image(src_jpg_image)
            pil_image = json_imageData_to_pil_image(src_jpg_image)

            try:
                pil_image.save(dst_image)
            except IOError:
                print("cannot convert", dst_image)

            imageData = pil_img_to_b64(Image.open(dst_image), format='str')

            # json 에 저장된 jpg 이미지 binary string 을 png 로 대치
            dst_json_file = open(dst_json, 'w')
            dst_json_obj = src_json_obj.copy()
            dst_json_obj['imagePath'] = filename + '.PNG'
            dst_json_obj['imageData'] = imageData
            json.dump(dst_json_obj, dst_json_file, indent=2)
            dst_json_file.close()


if __name__ == "__main__":
    # json 에 담긴 jpg 데이터정보를 png 로 뱌꿈
    # src 경로에 이미지파일 없이 json 만 있어야 작동하며 json 에서 imageData 를 뽑아내 dst 경로에 이미지를 함께 저장함
    src_dir = '/media/qisens/4tb3/navermap_detection/dataset/scribble_dataset/building_capture/json_to_png/jpg_dataset/'
    dst_dir = '/media/qisens/4tb3/navermap_detection/dataset/scribble_dataset/building_capture/json_to_png/png_dataset/'

    jpg_to_png(src_dir, dst_dir)
