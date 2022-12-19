import os
from PIL import Image, ImageDraw, ImageFont
import glob
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#load inference images and sort the list ascending order
inferenced_img_path= "./run/tree/resnet/inference_output"
origin_path= "./inference_test_img"
output_path = "./run/tree/resnet/comparison_inference_result"
if not os.path.exists(output_path):
    os.mkdir(output_path)

img1_ls = glob.glob(os.path.join(inferenced_img_path, "*.png"))
origin_ls = glob.glob(os.path.join(origin_path, "*.png"))

img1_ls.sort()
# img2_ls.sort()
origin_ls.sort()

#color chart bar dict
# color_bar = {"bg":(0, 0, 0), "tree & grass":(21, 134, 55), "yellow_grass":(0, 255, 43), "green_obj":(248, 255, 3), "ground":(137, 69, 18)}
color_bar = {"bg":(0, 0, 0), "tree":(21, 134, 55), "grass":(0, 255, 43), "roof":(248, 255, 3), "ground":(137, 69, 18), "obj":(70,255,255), "rail":(253, 21, 21)}
# color_bar = {"bg":(0,0,0), "flatroof":(0, 0, 255), "facility":(255, 0, 0), "rooftop":(255, 250, 0), 'gt':(239,0,255)}
number_of_inference = 2

for i in range(len(img1_ls)):
    #original image has different size comparing to others. So, I resized the original image.
    image_ls = [Image.open(origin_ls[i]),Image.open(img1_ls[i])]
    image_fixed_size = image_ls[1].size
    image_ls[0] = image_ls[0].resize(image_fixed_size)

    file_path , file_name = os.path.split(img1_ls[i])

    #make blank canvas and set font
    canvas = Image.new('RGB', (image_fixed_size[0]*number_of_inference + 300, image_fixed_size[1]+80), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("./arial.ttf", 27)
    font2 = ImageFont.truetype("./arial.ttf", 25)

    #text related variables
    x_offset=0
    text_x_offset= round(image_fixed_size[0]/2-40)
    text_y = image_fixed_size[1]+35
    text_ls = ["original","scribble"]

    #concatenate images and draw text below them
    for idx, im in enumerate(image_ls):
        canvas.paste(im, (x_offset, 20))
        draw.text((text_x_offset, text_y), text_ls[idx], font=font, fill=(0,0,0))
        x_offset+=image_fixed_size[0] + 10
        text_x_offset+=image_fixed_size[0]

    #display color chart bar
    bar_x = image_fixed_size[0]*number_of_inference
    bar_y_offset = 80
    for key in color_bar.keys():
        draw.rectangle([(bar_x + 50, bar_y_offset), (bar_x + 110, bar_y_offset + 30)], fill=color_bar[key])
        draw.text((bar_x + 120, bar_y_offset), key, font=font2, fill=(0,0,0))
        bar_y_offset += 50

    #save result images
    canvas.save(os.path.join(output_path, file_name))
