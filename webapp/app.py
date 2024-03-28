from flask import Flask, render_template, request, redirect, url_for, jsonify
from mmdet.apis import init_detector, inference_detector, DetInferencer
import os
import time
import cv2
from pycocotools import mask
import numpy
from mmdet.registry import VISUALIZERS
from math import pi
from PIL import Image
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Initialize your mmdetection model
config_file = 'C:/Users/peter/mmdetection/configs/AI4Food/configs/foodseg103coco.py'
checkpoint_file = 'C:/Users/peter/mmdetection/work_dirs/foodseg103coco/epoch_25.pth'
sticker_config = 'C:/Users/peter/mmdetection/configs/AI4Food/configs/stickerPre.py'
sticker_checkpoint = 'C:/Users/peter/mmdetection/work_dirs/stickerPre/epoch_4.pth'
# model = init_detector(config_file, checkpoint_file)
DIAMETER = 0.9
size = 640, 640
categories = ('candy', 'egg tart', 'french fries', 'chocolate', 'biscuit', 'popcorn', 'pudding', 'ice cream', 'cheese butter', 'cake', 'wine', 'milkshake', 'coffee', 'juice', 'milk', 'tea', 'almond', 'red beans', 'cashew', 'dried cranberries', 'soy', 'walnut', 'peanut', 'egg', 'apple', 'date', 'apricot', 'avocado', 'banana', 'stawberry', 'cherry', 'blueberry', 'raspberry', 'mango', 'olives', 'peach', 'lemon', 'pear', 'fig', 'pineapple', 'grape', 'kiwi', 'melon', 'orange', 'watermelon', 'steak', 'pork', 'chicken duck', 'sausage', 'fried meat', 'lamb', 'sauce', 'crab', 'fish', 'shellfish', 'shrimp', 'soup', 'bread', 'corn', 'hamburg', 'pizza', 'hanaki baozi', 'wonton dumplings', 'pasta', 'noodles', 'rice', 'pie', 'tofu', 'eggplant', 'potato', 'garlic', 'cauliflower', 'tomato', 'kelp', 'seaweed', 'spring onion', 'rape', 'ginger', 'okra', 'lettuce', 'pumpkin', 'cucumber', 'white radish', 'carrot', 'asparagus', 'bamboo shoots', 'broccoli', 'celery stick', 'cilantro mint', 'snow peas', 'cabbage', 'bean sprouts', 'onion', 'pepper', 'green beans', 'french beans', 'king oyster mushroom', 'shiitake', 'enoki mushroom', 'oyster mushroom', 'white button mushroom', 'salad', 'other ingredients')
cal_per_100g = {'candy': (450, 1), 'egg tart': (225, 0), 'french fries': (312, 1), 'chocolate': (418, 1), 'biscuit': (347, 1), 'popcorn': (460, 0), 'pudding': (300, 'ice cream': (200, 'cheese butter': (349, 'cake': (257, 'wine': (83, 'milkshake': (112, 'coffee': (20, 'juice': (54, 'milk': (42, 'tea': (10, 'almond': (529, 'red beans': (333, 'cashew': (553, 'dried cranberries': (308, 'soy': (446, 'walnut': (654, 'peanut': (567, 'egg': (155, 'apple': (52, 'date': (282, 'apricot': (48, 'avocado': (160, 'banana': (89, 'stawberry': (33, 'cherry': (50, 'blueberry': (57, 'raspberry': (53, 'mango': (60, 'olives': (115, 'peach': (39, 'lemon': (29, 'pear': (57, 'fig': (74, 'pineapple': (50, 'grape': (67, 'kiwi': (61, 'melon': (34, 'orange': (47, 'watermelon': (30, 'steak': (271, 'pork': (242, 'chicken duck': (239, 'sausage': (301, 'fried meat': (301, 'lamb': (294, 'sauce': (68, 'crab': (97, 'fish': (206, 'shellfish': (99, 'shrimp': (99, 'soup': (32, 'bread': (265, 'corn': (86, 'hamburg': (295, 'pizza': (266, 'hanaki baozi': (147, 'wonton dumplings': (284, 'pasta': (131, 'noodles': (138, 'rice': (130, 'pie': (237, 'tofu': (76, 'eggplant': (25, 'potato': (75, 'garlic': (149, 'cauliflower': (25, 'tomato': (19, 'kelp': (43, 'seaweed': (306, 'spring onion': (32, 'rape': (28, 'ginger': (80, 'okra': (33, 'lettuce': (15, 'pumpkin': (26, 'cucumber': (15, 'white radish': (16, 'carrot': (41, 'asparagus': (20, 'bamboo shoots': (27, 'broccoli': (34, 'celery stick': (14, 'cilantro mint': (23, 'snow peas': (42, 'cabbage': (25, 'bean sprouts': (30, 'onion': (40, 'pepper': (20, 'green beans': (31, 'french beans': (31, 'king oyster mushroom': (35, 'shiitake': (34, 'enoki mushroom': (37, 'oyster mushroom': (33, 'white button mushroom': (22, 'salad': (17, 'other ingredients': (0}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = str(int(time.time())) + '_' + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('inference', filename=filename))
    else:
        return redirect(request.url)

@app.route('/inference/<filename>')
def inference(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result = inference_image(filepath)
    return render_template('result.html', filename=filename, result=result)

def inference_image(image_path):

    start = time.time()
    im = Image.open(image_path) #Code from: https://stackoverflow.com/questions/9174338/programmatically-change-image-resolution
    im_resized = im.resize(size)
    im_resized.save(image_path, "JPEG")
    # Initialize the DetInferencer
    print("inference_detector: ", start)
    model1 = init_detector(config_file, checkpoint_file)
    pred = inference_detector(model1, image_path)

    print("DetInfer: ", time.time())
    inferencer = DetInferencer(model=config_file, weights=checkpoint_file, palette=None)
    result = inferencer(image_path, out_dir='static/out/', show=False)

    print("pred_ti_dict(): ", time.time() - start)
    pred_masks, pred_dict = pred_to_dict(pred, start)

    arealist = []

    print("areaList: ", time.time() - start)
    for i in range(len(pred_masks)):
        contours, _ = cv2.findContours(pred_masks[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        temp = [cv2.contourArea(contours[0]), pred_dict['scores'][i], pred_dict['labels'][i] + 1, categories[pred_dict['labels'][i]]]
        arealist.append(temp)

    print("ratio: ", time.time() - start)
    ratio = pixel_to_area(image_path, start)

    print("ratio use: ", time.time() - start)
    for i in arealist:
        tempArea = i[0] * ratio
        tempCalories = tempArea / 100
        tempCalories = tempCalories * cal_per_100g.get(categories[(i[2]-1)])
        i.append(tempArea)
        i.append(tempCalories)
        print(i)

    print("calories start: ", time.time() - start)
    # areaCalList = area_to_cal(arealist)
    # for i in arealist:
    #     calories = i * 0.01
    #     calories = calories  * cal_per_100g.get(categories[(i[2]-1)])
    #     arealist.append(calories)


    print("return: ", time.time() - start)
    return arealist

# Code form sizing_module.py provided by Chris Moorhead, modfied for numpy output
def pred_to_dict(pred, start):
    print("pred to dict start: ", time.time() - start)
    pred_masks = pred.pred_instances.masks.clone().detach().cpu().numpy()
    pred_scores = pred.pred_instances.scores.clone().detach().cpu().numpy()
    pred_labels = pred.pred_instances.labels.clone().detach().cpu().numpy()
    pred_dict = {"masks": pred_masks, "scores": pred_scores, "labels": pred_labels}

    # topX = Nmaxelements(pred_scores, len(pred_dict['masks']))
    # topX_index = []
    # for i in topX:
    #     list1 = list(map(float, pred_scores))
    #     topX_index.append(list1.index(i))


    print("predtodict numpy start: ", time.time() - start)
    pred_masks = []
    for i in pred_dict['masks']:
        temp = []
        for j in i:
            temp1 = list(map(int, j))

            if len(temp1) != 0:
                temp1 = numpy.array(temp1, dtype=numpy.uint8)
                temp.append(temp1)

        if len(temp) != 0:
            temp = numpy.array(temp)
            pred_masks.append(temp)

    pred_masks = numpy.array(pred_masks)

    print("predtodict return: ", time.time() - start)
    return pred_masks, pred_dict

def pixel_to_area(img_path, start):
    print("pixeltoarea start: ", time.time() - start)
    model2 = init_detector(sticker_config, sticker_checkpoint)
    pred_sticker = inference_detector(model2, img_path)
    pred_mask_sticker, pred_dict_sticker = pred_to_dict(pred_sticker, start)
    sticker_area = (DIAMETER / 2)**2 * pi
    pixel_to_area_ratio = 0
    arealistSticker = []

    print("pixeltoarea contour start: ", time.time() - start)
    for i in range(len(pred_mask_sticker)):
        contours, _ = cv2.findContours(pred_mask_sticker[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        temp = [cv2.contourArea(contours[0]), pred_dict_sticker['scores'][i], pred_dict_sticker['labels'][i]] # categories[int(pred_dict_sticker['labels'][i])]
        arealistSticker.append(temp)

    max_index = arealistSticker[1].index(max(arealistSticker[1]))

    if arealistSticker[max_index][2] == 0:
        pixel_to_area_ratio = sticker_area / arealistSticker[0][max_index]

    print("pixeltoarea return: ", time.time() - start)
    return pixel_to_area_ratio

def area_to_cal(maskList):
    for i in maskList:
        # print(type(i))
        # tempNum = float(i[4])
        calories = i[4] / 100 * cal_per_100g.get(categories[(i[2]-1)])
        maskList.append(calories)

    return maskList

#code from geeksforgeeks
def Nmaxelements(list1, N):
    final_list = []
    list1 = list(map(float, list1))

    for i in range(0, N):
        max1 = 0

        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j]

        list1.remove(max1)
        final_list.append(max1)

    return final_list

if __name__ == '__main__':
    app.run(debug=True)
