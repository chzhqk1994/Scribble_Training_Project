import os
import cv2
import time
import math
import numpy as np
from PIL import Image
from torchvision import transforms
from dataloaders.utils_tree import decode_segmap
from dataloaders import custom_transforms as tr
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from pyproj import proj, transform
from geopy import distance

global grad_seg
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

proj_wgs84 = proj.Proj('epsg:4326')
proj_naver = proj.Proj('epsg:5179')


class Scribble:
    def __init__(self, gpu_num, model_path, backbone, n_class, dataset_name='tree', num_workers=1):
        self.gpu_num = gpu_num
        self.model_path = model_path
        self.backbone = backbone
        self.n_class = n_class
        self.dataset_name = dataset_name
        self.num_workers = num_workers

        self.model = None
        self.init_gpu()
        self.composed_transforms = transforms.Compose([
            tr.NormalizeImage(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensorImage()])

    def init_gpu(self):
        # Define network
        self.model = DeepLab(num_classes=self.n_class,
                        backbone=self.backbone,
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)

        # Using cuda
        self.model = torch.nn.DataParallel(self.model, device_ids=[self.gpu_num])
        patch_replication_callback(self.model)
        self.model = self.model.cuda()

        # load checkpoint model
        if not os.path.isfile(self.model_path):
            raise RuntimeError("=> no checkpoint found at '{}'".format(self.model_path))
        checkpoint = torch.load(self.model_path)
        self.model.module.load_state_dict(checkpoint['state_dict'])
        best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {}) best_pred {}"
              .format(self.model_path, checkpoint['epoch'], best_pred))

        self.model.eval()

    def inference(self, r2cnn_bbox_img):
        # convert from openCV2 to PIL
        color_coverted = cv2.cvtColor(r2cnn_bbox_img, cv2.COLOR_BGR2RGB)
        pil_r2cnn_bbox_img = Image.fromarray(color_coverted)
        image_size = r2cnn_bbox_img.size

        image = self.composed_transforms(pil_r2cnn_bbox_img.convert('RGB')).unsqueeze(0)
        image = image.cuda()
        start = time.time()

        output = self.model(image)
        print('inference time:', time.time() - start)
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)

        # visualize prediction
        segmap = decode_segmap(pred[0], self.dataset_name) * 255
        segmap = segmap.astype(np.uint8)
        segimg = Image.fromarray(segmap, 'RGB')

        segmap = cv2.cvtColor(segmap, cv2.COLOR_RGB2BGR)
        building_img_width, building_img_height = segimg.size

        # origin_tile_img_path = "./13_45_957564.3912121656_1949308.4513075682_Navermap.png"  # tiling 된 이미지 한 장
        # meter_constant = calculate_meter_constant(origin_tile_img_path)  # 미터상수 계산 (zoom level 변동없으면 미터상수는 0.1171)
        meter_constant = 0.1171  # 640px = 75m, 1px = 0.1171m

        segmap, output = self.get_cropped_segmap(r2cnn_bbox_img, segmap, output)  # 이미지에서 건물범위, flat 부분만 받아옴

        building_area, flat_ratio = self.calculate_flat_area(r2cnn_bbox_img, output, building_img_width, building_img_height, meter_constant)  # 건물면적 및 flat 비율 계산
        print("flat 비율 : ", flat_ratio)
        print("전체 건물 면적 (meter) : {} meter^2".format(building_area))
        print("flat 면적 (meter) : {} meter^2".format(building_area * flat_ratio))

        # r2cnn_bbox_img : original
        # segmap : scribble result
        # output : only flat area
        # roof_area = np.hstack([r2cnn_bbox_img, segmap, output])  # original, segmap, flatarea 붙인 이미지
        return segmap, output

    def get_pix_from_gps(self, centerLng, centerLat, targetLng, targetLat, width, height,  zoom=20):
        parallelMultiplier = math.cos(centerLat * math.pi / 180)
        degreesPerPixelX = 360 / math.pow(2, zoom + 8)
        degreesPerPixelY = 360 / math.pow(2, zoom + 8) * parallelMultiplier
        pY = (centerLat - targetLat) / degreesPerPixelY + height / 2
        pX = (targetLng - centerLng) / degreesPerPixelX + width / 2

        return int(pX), int(pY)

    def get_gps_from_pix(self, centerLng, centerLat, zoom, px, py, width, height):
        parallelMultiplier = math.cos(centerLat * math.pi / 180)
        degreesPerPixelX = 360 / math.pow(2, zoom + 8)
        degreesPerPixelY = 360 / math.pow(2, zoom + 8) * parallelMultiplier
        pointLat = centerLat - degreesPerPixelY * (py - height / 2)
        pointLng = centerLng + degreesPerPixelX * (px - width / 2)

        return (pointLng, pointLat)

    def calculate_meter_constant(self, origin_tile_img_path):
        origin_img_path = origin_tile_img_path
        naver_coord = (float(origin_img_path.split('/')[-1].split('_')[2]), float(origin_img_path.split('/')[-1].split('_')[3]))
        origin_img = cv2.imread(origin_img_path)
        origin_img_height, origin_img_width, _ = origin_img.shape

        # lat, lon = transform(proj_naver, proj_wgs84, 1949308.4513075682, 957564.3912121656)
        center_lat, center_lon = transform(proj_naver, proj_wgs84, naver_coord[1], naver_coord[0])  # 이미지 중심좌표
        print(center_lat, center_lon)

        # 이미지의 가로세로 거리 계산
        nw_lon, nw_lat = self.get_gps_from_pix(center_lon, center_lat, 20, 0, 0, origin_img_width,
                                          origin_img_height)  # 좌상단
        ne_lon, ne_lat = self.get_gps_from_pix(center_lon, center_lat, 20, origin_img_width, 0, origin_img_width,
                                          origin_img_height)  # 우상단
        sw_lon, sw_lat = self.get_gps_from_pix(center_lon, center_lat, 20, 0, origin_img_height, origin_img_width,
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

    def calculate_flat_area(self, building_image_cv2, flat_image_cv2, building_img_width, building_img_height, meter_constant):
        building_only_area = cv2.countNonZero(cv2.cvtColor(building_image_cv2, cv2.COLOR_BGR2GRAY))  # 건물에만 해당하는 픽셀 수
        building_only_area_ratio = building_only_area / (building_img_width * building_img_height)  # 건물 이미지에서 건물면적의 비율
        building_img_whole_area_meter = (building_img_width * meter_constant) * (
                    building_img_height * meter_constant)  # 건물 이미지의 전체 meter 면적(bg 포함)
        building_only_area_meter = building_img_whole_area_meter * building_only_area_ratio  # 건물의 면적만을 계산한 결과 값 (meter)
        print(building_only_area_ratio, building_only_area_meter)

        flat_area = cv2.countNonZero(cv2.cvtColor(flat_image_cv2, cv2.COLOR_BGR2GRAY))  # inference 결과 중 flat 에만 해당하는 픽셀 수 (건물에 해당하는 픽셀만 남아있는 상태)
        flat_ratio = flat_area / building_only_area

        return building_only_area_meter, flat_ratio

    def get_cropped_segmap(self, building_image_cv2, segmap, output):
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


if __name__ == '__main__':
    gpu_num = 0
    model_path = '../pytorch/pytorch-deeplab_v3_plus/run/tree/resnet/model_best.pth.tar_withwall'
    backbone = 'resnet'
    n_class = 13
    dataset = 'tree'
    input_img_dir = '../pytorch/pytorch-deeplab_v3_plus/scribble_test_area/r2cnn_bbox/'
    output_img_dir = ''
    scribble_obj = Scribble(gpu_num, model_path, backbone, n_class, dataset)

    total = 0
    for path, dirs, files in os.walk(input_img_dir):
        for file in files:
            total += 1
            current_img_file = cv2.imread(os.path.join(input_img_dir, file))
            segmap, flat_only_img = scribble_obj.inference(current_img_file)
