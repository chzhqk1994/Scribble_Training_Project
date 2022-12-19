import numpy as np
import cv2
import os

def get_cropped_segmap(origin_img):
    org_h, org_w, _ = origin_img.shape

    masked = origin_img.copy()
    # 원본이미지의 RGB가 (0, 0, 0) 인 부분을 output 이미지에서 제거
    for y in range(0, org_h):
        for x in range(0, org_w):
            if not origin_img[y][x].all():  # building_image_cv2[x][y] == rgb 값
                masked[y][x] = np.random.randint(low=0, high=255, size=(1, 1, 3), dtype=np.uint8)

    return masked


if __name__ == '__main__':
    org_dir = '/media/qisens/4tb3/navermap_detection/dataset/scribble_dataset/building_capture/wall_to_concrete/augment/poly_crop_data_noise/original_png/'
    output_dir = '/media/qisens/4tb3/navermap_detection/dataset/scribble_dataset/building_capture/wall_to_concrete/augment/poly_crop_data_noise/noise/'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for root, dirs, files in os.walk(org_dir):
        for file in files:
            #print(file)
            origin_img = cv2.imread(os.path.join(root, file))
            output = get_cropped_segmap(origin_img)
            output_img_path = os.path.join(output_dir, file)
            print(output_img_path)
            cv2.imwrite(output_img_path, output)

            # cv2.imshow("origin_img", origin_img)
            # cv2.imshow("output", output)
            # cv2.waitKey(0)


