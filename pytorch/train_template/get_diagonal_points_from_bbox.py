import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement
import cv2
import random


def build_xml_arch(parsed_xml):
    label = ''
    x0 = 0
    x1 = 0
    x2 = 0
    x3 = 0
    y0 = 0
    y1 = 0
    y2 = 0
    y3 = 0

    # label_dict = {"bg":(0,0,0), "tree":(1,1,1), "roof":(2,2,2), "grass":(3,3,3), "ground":(4,4,4), "obj":(5,5,5), "rail":(6,6,6)}
    # tree_cnt = 0
    # roof_cnt = 0
    # grass_cnt = 0
    # ground_cnt = 0
    # obj_cnt = 0
    # rail_cnt = 0

    label_dict = {"bg": (0, 0, 0), "goodroof": (1, 1, 1), "facility": (2, 2, 2), "rooftop": (3, 3, 3)}
    rooftop_cnt = 0
    facility_cnt = 0
    goodroof_cnt = 0

    saved_file = False
    saved_file_cnt = 0
    all_object_coordinates = []

    annotation = Element('annotation')
    folder = SubElement(annotation, 'folder')
    folder.text = parsed_xml['folder']

    filename = SubElement(annotation, 'filename')
    filename.text = parsed_xml['filename']

    path = SubElement(annotation, 'path')
    path.text = parsed_xml['path']

    source = SubElement(annotation, 'source')
    database = SubElement(source, 'database')
    database.text = 'Unknown'

    size = SubElement(annotation, 'size')
    width = SubElement(size, 'width')
    width.text = str(640)
    height = SubElement(size, 'height')
    height.text = str(640)
    depth = SubElement(size, 'depth')
    depth.text = str(3)

    segmented = SubElement(annotation, 'segmented')
    segmented.text = '0'

    width, height = 640, 640
    img = np.zeros((height, width, 3), np.uint8)
    img[:, :] = [255, 255, 255]


    for obj in parsed_xml['objects']:
        # xml 태그가 robndbox 인 경우
        if not obj.find('robndbox') is None:
            label = obj.find('name').text
            cx = obj.find('./robndbox/cx').text
            cy = obj.find('./robndbox/cy').text
            width = obj.find('./robndbox/w').text
            height = obj.find('./robndbox/h').text
            angle = obj.find('./robndbox/angle').text
            try:
                x0 = obj.find('./robndbox/x0').text
                x1 = obj.find('./robndbox/x1').text
                x2 = obj.find('./robndbox/x2').text
                x3 = obj.find('./robndbox/x3').text
                y0 = obj.find('./robndbox/y0').text
                y1 = obj.find('./robndbox/y1').text
                y2 = obj.find('./robndbox/y2').text
                y3 = obj.find('./robndbox/y3').text
            except AttributeError:
                print("누락")
                cx = float(cx)
                cy = float(cy)
                w = float(width)
                h = float(height)
                angle = float(angle)


                cx, cy, w, h, angle = str(cx), str(cy), str(w), str(h), str(angle)

                # str() >> 좌표들을 xml로 저장하기 위해선 text 로 바꿔야 하기 때문에 str 로 묶음
                # int() >> 좌표를 R2CNN 에선 정수형으로 변환하기 때문에 int 로 묶음
                # float() >> 실수형 좌표를 정수로 바꾸기 위해 float 으로 먼저 형변환을 해줌 >> 왜 해야하는거지
            x0, x1, x2, x3, y0, y1, y2, y3 = str(int(float(x0))), str(int(float(x1))), str(int(float(x2))), str(
                int(float(x3))), str(int(float(y0))), str(int(float(y1))), str(int(float(y2))), str(int(float(y3)))

        # xml 태그가 bndbox 인 경우
        if not obj.find('bndbox') is None:
            # robndbox 를 다루기 때문에 중심, 세로, 높이를 계산해야 하지만 bndbox 는 회전각이 0도 이기때문에 max min 값을 그대로 사용가능
            # 중심, 세로, 높이를 중요시 하는 경우 0을 계산해서 수정.
            label = obj.find('name').text
            x0 = obj.find('./bndbox/xmin').text
            x1 = obj.find('./bndbox/xmax').text
            x2 = obj.find('./bndbox/xmax').text
            x3 = obj.find('./bndbox/xmin').text
            y0 = obj.find('./bndbox/ymin').text
            y1 = obj.find('./bndbox/ymin').text
            y2 = obj.find('./bndbox/ymax').text
            y3 = obj.find('./bndbox/ymax').text
            x0, x1, x2, x3, y0, y1, y2, y3 = str(int(float(x0))), str(int(float(x1))), str(int(float(x2))), str(
                int(float(x3))), str(int(float(y0))), str(int(float(y1))), str(int(float(y2))), str(int(float(y3)))

        # if label == "tree" or label == "roof" or label == "grass" or label == "ground" or label == "obj" or label == "rail":
        #     if label == "tree":
        #         tree_cnt += 1
        #         label_color = label_dict["tree"]
        #     elif label == "roof":
        #         roof_cnt += 1
        #         label_color = label_dict["roof"]
        #     elif label == "grass":
        #         grass_cnt += 1
        #         label_color = label_dict["grass"]
        #     elif label == "ground":
        #         ground_cnt += 1
        #         label_color = label_dict["ground"]
        #     elif label == "obj":
        #         obj_cnt += 1
        #         label_color = label_dict["obj"]
        #     elif label == "rail":
        #         rail_cnt += 1
        #         label_color = label_dict["rail"]

        if label == "rooftop" or label == "facility" or label == "goodroof":
            if label == "rooftop":
                rooftop_cnt += 1
                label_color = label_dict["rooftop"]
            elif label == "facility":
                facility_cnt += 1
                label_color = label_dict["facility"]
            elif label == "goodroof":
                goodroof_cnt += 1
                label_color = label_dict["goodroof"]

            all_object_coordinates.append("{},{}/{},{}/{},{}/{},{}".format(x0, y0, x1, y1, x2, y2, x3, y3))
            cv2.line(img, (int(x0), int(y0)), (int(x2), int(y2)), label_dict[label], thickness=2)

    arr_x = [ hor_coord for hor_coord in range(0, 640)]
    arr_y = [ ver_coord for ver_coord in range(0, 640)]

    for coord in all_object_coordinates:
        each_coord = coord.split('/')
        x0, y0 = int(each_coord[0].split((','))[0]), int(each_coord[0].split((','))[1])
        x1, y1 = int(each_coord[1].split((','))[0]), int(each_coord[1].split((','))[1])
        x2, y2 = int(each_coord[2].split((','))[0]), int(each_coord[2].split((','))[1])
        x3, y3 = int(each_coord[3].split((','))[0]), int(each_coord[3].split((','))[1])

        x_length_for_0_to_2, x_length_for_1_to_3 = abs(x2 - x0), abs(x3 - x1)
        y_length_for_0_to_2, y_length_for_1_to_3 = abs(y2 - y0), abs(y3 - y1)

        if x_length_for_0_to_2 > x_length_for_1_to_3:
            if x0 > x2:
                prev_x, next_x = x2, x0
            else:
                prev_x, next_x = x0, x2
        else:
            if x1 > x3:
                prev_x, next_x = x3, x1
            else:
                prev_x, next_x = x1, x3
        if y_length_for_0_to_2 > y_length_for_1_to_3:
            if y0 > y2:
                prev_y, next_y = y2, y0
            else:
                prev_y, next_y = y0, y2
        else:
            if y1 > y3:
                prev_y, next_y = y3, y1
            else:
                prev_y, next_y = y1, y3

        for x_ in range(prev_x, next_x):
            try:
                arr_x.remove(x_)
            except:
                continue
        for y_ in range(prev_y, next_y):
            try:
                arr_y.remove(y_)
            except:
                continue

    if len(arr_x) >= 2:
        random.shuffle(arr_x)
        rand_x = random.sample(arr_x, 2)
    else:
        rand_x = arr_x
    if len(arr_y) >= 2:
        random.shuffle(arr_y)
        rand_y = random.sample(arr_y, 2)
    else:
        rand_y = arr_y

    # for idx, arr_x_ in enumerate(arr_x):
    #     if idx % 15 == 0:
    #         cv2.line(img, (arr_x_, 0), (arr_x_, 640), label_dict["bg"], thickness=1)
    # for idx, arr_y_ in enumerate(arr_y):
    #     if idx % 15 == 0:
    #         cv2.line(img, (0, arr_y_), (640, arr_y_), label_dict["bg"], thickness=1)

    for idx, arr_x_ in enumerate(rand_x):
        cv2.line(img, (arr_x_, 0), (arr_x_, 640), label_dict["bg"], thickness=2)
    for idx, arr_y_ in enumerate(rand_y):
        cv2.line(img, (0, arr_y_), (640, arr_y_), label_dict["bg"], thickness=2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(xml_parse["output_dir"], xml_parse["filename"]) + ".png", gray)
    saved_file = True
    print(xml_parse["filename"] + " is completed")

    if saved_file:
        saved_file_cnt += 1
    return goodroof_cnt, facility_cnt, rooftop_cnt, saved_file_cnt
    # return tree_cnt, roof_cnt, grass_cnt, ground_cnt, obj_cnt, rail_cnt, saved_file_cnt




if __name__ == "__main__":

    anno_dir ="/media/qisens/2tb1/goodroof_solarpanel_parkinglot_rooftop_facility_rotation/test/annotations"
    output_dir="/media/qisens/2tb1/goodroof_solarpanel_parkinglot_rooftop_facility_rotation/test/scribble_img_new"
    cnt_dir = "/media/qisens/2tb1/goodroof_solarpanel_parkinglot_rooftop_facility_rotation/test"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    goodroof_cnt = 0
    facility_cnt = 0
    rooftop_cnt = 0
    saved_file_cnt = 0

    # tree_cnt = 0
    # roof_cnt = 0
    # grass_cnt = 0
    # ground_cnt = 0
    # obj_cnt = 0
    # rail_cnt = 0
    # saved_file_cnt = 0

    for path, dirs, files in os.walk(anno_dir):
        for file in files:
            file_path = os.path.join(path,file)
            xml_parse = {'folder': '', 'filename': '', 'path': '', 'objects': [], 'output_dir':''}
            tree = ET.parse(file_path)
            obj_name_elements = tree.findall('./object')
            if len(obj_name_elements) == 0:  # object 가 하나도 없다면 return None
                print(1)
            xml_parse['objects'] = obj_name_elements
            bndbox_name_elements = tree.findall('*/bndbox')
            if len(bndbox_name_elements) == 0:
                pass
            robndbox_name_elements = tree.findall('*/robndbox')
            if len(robndbox_name_elements) == 0:
                pass

            xml_parse['folder'] = tree.getroot().find('folder').text
            # xml_parse['filename'] = tree.getroot().find('filename').text
            xml_parse['filename'] = file
            xml_parse['path'] = tree.getroot().find('path').text
            xml_parse['output_dir'] = output_dir

            # tree, roof, grass, ground, obj, rail, save_file = build_xml_arch(xml_parse)
            # tree_cnt += tree
            # roof_cnt += roof
            # grass_cnt += grass
            # ground_cnt += ground
            # obj_cnt += obj
            # rail_cnt += rail
            # saved_file_cnt += save_file

            goodroof, facility, rooftop, save_file = build_xml_arch(xml_parse)
            goodroof_cnt += goodroof
            facility_cnt += facility
            rooftop_cnt += rooftop
            saved_file_cnt += save_file

    # with open(os.path.join(cnt_dir, "cnt.txt"), 'w') as f:
    #     f.write("total  --- goodroof : [{}] | facility : [{}] | rooftop : [{}] | saved_file : [{}]".format(goodroof_cnt, facility_cnt, rooftop_cnt, saved_file_cnt))

    print("total  --- goodroof : [{}] | facility : [{}] | rooftop : [{}] | saved_file : [{}]".format(goodroof_cnt, facility_cnt, rooftop_cnt, saved_file_cnt))
    # print("total  --- tree : [{}] | roof : [{}] | grass : [{}] | ground : [{}] | obj : [{}] | rail : [{}]  saved_file : [{}]".format(tree_cnt,
    #                                                                                                  roof_cnt,
    #                                                                                                  grass_cnt, ground_cnt, obj_cnt, rail_cnt,
    #                                                                                                  saved_file_cnt))
    #




