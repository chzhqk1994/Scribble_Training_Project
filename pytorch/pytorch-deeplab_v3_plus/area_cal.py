import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from os.path import join, isdir

from mypath import Path
from dataloaders import make_data_loader
from dataloaders.custom_transforms import denormalizeimage
from dataloaders.utils_tree import decode_segmap
from dataloaders import custom_transforms as tr
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.saver import Saver
import time
import multiprocessing
import torchvision
from DenseCRFLoss import DenseCRFLoss
import math
import cv2
from pyproj import proj, transform
from geopy import distance

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

global grad_seg
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

proj_wgs84 = proj.Proj('epsg:4326')
proj_naver = proj.Proj('epsg:5179')


def get_pix_from_gps(centerLng, centerLat, targetLng, targetLat, width, height,  zoom=20):
    parallelMultiplier = math.cos(centerLat * math.pi / 180)
    degreesPerPixelX = 360 / math.pow(2, zoom + 8)
    degreesPerPixelY = 360 / math.pow(2, zoom + 8) * parallelMultiplier
    pY = (centerLat - targetLat) / degreesPerPixelY + height / 2
    pX = (targetLng - centerLng) / degreesPerPixelX + width / 2

    return int(pX), int(pY)


def get_gps_from_pix(centerLng, centerLat, zoom, px, py, width, height):
    parallelMultiplier = math.cos(centerLat * math.pi / 180)
    degreesPerPixelX = 360 / math.pow(2, zoom + 8)
    degreesPerPixelY = 360 / math.pow(2, zoom + 8) * parallelMultiplier
    pointLat = centerLat - degreesPerPixelY * (py - height / 2)
    pointLng = centerLng + degreesPerPixelX * (px - width / 2)

    return (pointLng, pointLat)


def calculate_meter_constant(origin_tile_img_path):
    origin_img_path = origin_tile_img_path
    naver_coord = (float(origin_img_path.split('/')[-1].split('_')[2]), float(origin_img_path.split('/')[-1].split('_')[3]))
    origin_img = cv2.imread(origin_img_path)
    origin_img_height, origin_img_width, _ = origin_img.shape

    # lat, lon = transform(proj_naver, proj_wgs84, 1949308.4513075682, 957564.3912121656)
    center_lat, center_lon = transform(proj_naver, proj_wgs84, naver_coord[1], naver_coord[0])  # 이미지 중심좌표
    print(center_lat, center_lon)

    # 이미지의 가로세로 거리 계산
    nw_lon, nw_lat = get_gps_from_pix(center_lon, center_lat, 20, 0, 0, origin_img_width,
                                      origin_img_height)  # 좌상단
    ne_lon, ne_lat = get_gps_from_pix(center_lon, center_lat, 20, origin_img_width, 0, origin_img_width,
                                      origin_img_height)  # 우상단
    sw_lon, sw_lat = get_gps_from_pix(center_lon, center_lat, 20, 0, origin_img_height, origin_img_width,
                                      origin_img_height)  # 좌하단

    width_meter = distance.distance((nw_lat, nw_lon), (ne_lat, ne_lon)).meters
    height_meter = distance.distance((nw_lat, nw_lon), (sw_lat, sw_lon)).meters
    # print('NW : ', nw_lat, nw_lon)
    # print('NE : ', ne_lat, ne_lon)
    # print('SW : ', sw_lat, sw_lon)
    # print("width in Meter : ", width_meter)
    # print("height in Meter : ", height_meter)
    meter_constant = width_meter / origin_img_width
    return meter_constant


def calculate_flat_area(building_image_cv2, flat_image_cv2, building_img_width, building_img_height, meter_constant):
    building_only_area = cv2.countNonZero(cv2.cvtColor(building_image_cv2, cv2.COLOR_BGR2GRAY))  # 건물에만 해당하는 픽셀 수
    building_only_area_ratio = building_only_area / (building_img_width * building_img_height)  # 건물 이미지에서 건물면적의 비율
    building_img_whole_area_meter = (building_img_width * meter_constant) * (
                building_img_height * meter_constant)  # 건물 이미지의 전체 meter 면적(bg 포함)
    building_only_area_meter = building_img_whole_area_meter * building_only_area_ratio  # 건물의 면적만을 계산한 결과 값 (meter)
    print(building_only_area_ratio, building_only_area_meter)

    flat_area = cv2.countNonZero(cv2.cvtColor(flat_image_cv2, cv2.COLOR_BGR2GRAY))  # inference 결과 중 flat 에만 해당하는 픽셀 수 (건물에 해당하는 픽셀만 남아있는 상태)
    flat_ratio = flat_area / building_only_area

    return building_only_area_meter, flat_ratio


# flat_area = cv2.countNonZero(
#     cv2.cvtColor(output, cv2.COLOR_BGR2GRAY))  # inference 결과 중 flat 에만 해당하는 픽셀 수 (건물에 해당하는 픽셀만 남아있는 상태)
# flat_ratio = (flat_area / building_only_area)
# print("flat 비율 : ", flat_ratio)
#
# print("전체 건물 면적 (meter) : {} meter^2".format(building_only_area_meter))
# print("flat 면적 (meter) : {} meter^2".format(building_only_area_meter * flat_ratio))


def get_cropped_segmap(building_image_cv2, segmap, output):
    building_image_width, building_image_height, _ = building_image_cv2.shape

    # create NumPy arrays from the boundaries
    lower = np.array([0, 255, 0], dtype="uint8")  # [0, 255, 0] == color of flat
    upper = np.array([0, 255, 0], dtype="uint8")  # [0, 255, 0] == color of flat
    # find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(segmap, lower, upper)
    output = cv2.bitwise_and(segmap, segmap, mask=mask)

    # 원본이미지의 RGB가 (0, 0, 0) 인 부분을 output 이미지에서 제거
    for y in range(0, building_image_height):
        for x in range(0, building_image_width):
            if not building_image_cv2[x][y].all():  # building_image_cv2[x][y] == rgb 값
                segmap[x][y] = [0, 0, 0]  # 건물만 남긴 이미지
                output[x][y] = [0, 0, 0]  # 건물에서 flat 만 남긴 이미지

    return segmap, output


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Inference")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--n_class', type=int, default=7)
    parser.add_argument('--dataset', type=str, default='tree')
    parser.add_argument('--crop_size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--no_cuda', action='store_true', default=
    False, help='disables CUDA training')
    # checking point
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='put the path to checkpoint if needed')
    # rloss options
    parser.add_argument('--rloss-weight', type=float,
                        metavar='M', help='densecrf loss')
    parser.add_argument('--rloss-scale', type=float, default=0.5,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb', type=float, default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy', type=float, default=80.0,
                        help='DenseCRF sigma_xy')

    # output directory
    parser.add_argument('--output_directory', type=str,
                        help='output directory')

    # input image
    parser.add_argument('--image_path', type=str, default='./misc/test.png',
                        help='input image path')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Define Dataloader
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    print(args)

    # Define network
    model = DeepLab(num_classes=args.n_class,
                    backbone=args.backbone,
                    output_stride=16,
                    sync_bn=False,
                    freeze_bn=False)

    # Using cuda
    if not args.no_cuda:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        patch_replication_callback(model)
        model = model.cuda()

    # load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if args.cuda:
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    best_pred = checkpoint['best_pred']
    print("=> loaded checkpoint '{}' (epoch {}) best_pred {}"
          .format(args.checkpoint, checkpoint['epoch'], best_pred))

    model.eval()

    composed_transforms = transforms.Compose([
        # tr.FixScaleCropImage(crop_size=args.crop_size),
        tr.NormalizeImage(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensorImage()])

    total = 0
    for path, dirs, files in os.walk(args.image_path):
        for file in files:
            total += 1

            image_path = os.path.join(path, file)
            image_ = Image.open(image_path)
            image_size = image_.size
            building_image_cv2 = cv2.cvtColor(np.array(image_), cv2.COLOR_RGB2BGR)

            image = composed_transforms(image_.convert('RGB')).unsqueeze(0)
            image_cpu = image
            if not args.no_cuda:
                image = image.cuda()
            start = time.time()

            output = model(image)
            print('inference time:', time.time() - start)
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            # visualize prediction
            segmap = decode_segmap(pred[0], args.dataset) * 255
            segmap = segmap.astype(np.uint8)
            segimg = Image.fromarray(segmap, 'RGB')

            segmap = cv2.cvtColor(segmap, cv2.COLOR_RGB2BGR)
            building_img_width, building_img_height = segimg.size

            origin_tile_img_path = "./run/tree/resnet/13_45_957564.3912121656_1949308.4513075682_Navermap.png"  # tiling 된 이미지
            meter_constant = calculate_meter_constant(origin_tile_img_path)  # 미터상수 계산 (zoom level 변동없으면 미터상수는 0.1171)
            # meter_constant = 0.1171  # 640px = 75m, 1px = 0.1171m

            segmap, output = get_cropped_segmap(building_image_cv2, segmap, output)  # 이미지에서 건물범위, flat 부분만 받아옴

            building_area, flat_ratio = calculate_flat_area(building_image_cv2, output, building_img_width, building_img_height, meter_constant)  # 건물면적 및 flat 비율 계산
            print("flat 비율 : ", flat_ratio)
            print("전체 건물 면적 (meter) : {} meter^2".format(building_area))
            print("flat 면적 (meter) : {} meter^2".format(building_area * flat_ratio))

            roof_area = np.hstack([building_image_cv2, segmap, output])
            cv2.imshow("images", roof_area)
            cv2.waitKey(0)


            if args.output_directory is not None:
                if not isdir(args.output_directory):
                    os.makedirs(args.output_directory)
                segimg = segimg.resize(image_size)
                segimg.save(os.path.join(args.output_directory, os.path.split(image_path)[-1]))
            else:
                plt.figure()
                plt.imshow(segimg)
                plt.show()


if __name__ == "__main__":
    main()
