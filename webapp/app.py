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

# Relevant model/checkpoint folders
config_file = 'C:/Users/peter/mmdetection/configs/AI4Food/configs/foodseg103coco.py'
checkpoint_file = 'C:/Users/peter/mmdetection/work_dirs/foodseg103coco/epoch_25.pth'
sticker_config = 'C:/Users/peter/mmdetection/configs/AI4Food/configs/stickerPre.py'
sticker_checkpoint = 'C:/Users/peter/mmdetection/work_dirs/stickerPre/epoch_4.pth'

#Calibration sticker diameter, image size, categories, & calories for each category
DIAMETER = 0.9
size = 640, 640
categories = ('candy', 'egg tart', 'french fries', 'chocolate', 'biscuit', 'popcorn', 'pudding', 'ice cream', 'cheese butter', 'cake', 'wine', 'milkshake', 'coffee', 'juice', 'milk', 'tea', 'almond', 'red beans', 'cashew', 'dried cranberries', 'soy', 'walnut', 'peanut', 'egg', 'apple', 'date', 'apricot', 'avocado', 'banana', 'stawberry', 'cherry', 'blueberry', 'raspberry', 'mango', 'olives', 'peach', 'lemon', 'pear', 'fig', 'pineapple', 'grape', 'kiwi', 'melon', 'orange', 'watermelon', 'steak', 'pork', 'chicken duck', 'sausage', 'fried meat', 'lamb', 'sauce', 'crab', 'fish', 'shellfish', 'shrimp', 'soup', 'bread', 'corn', 'hamburg', 'pizza', 'hanaki baozi', 'wonton dumplings', 'pasta', 'noodles', 'rice', 'pie', 'tofu', 'eggplant', 'potato', 'garlic', 'cauliflower', 'tomato', 'kelp', 'seaweed', 'spring onion', 'rape', 'ginger', 'okra', 'lettuce', 'pumpkin', 'cucumber', 'white radish', 'carrot', 'asparagus', 'bamboo shoots', 'broccoli', 'celery stick', 'cilantro mint', 'snow peas', 'cabbage', 'bean sprouts', 'onion', 'pepper', 'green beans', 'french beans', 'king oyster mushroom', 'shiitake', 'enoki mushroom', 'oyster mushroom', 'white button mushroom', 'salad', 'other ingredients')
cal_per_100g = {'candy': (450, 1), 'egg tart': (225, 0), 'french fries': (312, 1), 'chocolate': (418, 1), 'biscuit': (347, 1), 'popcorn': (460, 0), 'pudding': (300, 1), 'ice cream': (200, 1), 'cheese butter': (349, 2), 'cake': (257, 1), 'wine': (83, 0), 'milkshake': (112, 1), 'coffee': (20, 0), 'juice': (54, 0), 'milk': (42, 0), 'tea': (10, 0), 'almond': (529, 1), 'red beans': (333, 1), 'cashew': (553, 1), 'dried cranberries': (308, 1), 'soy': (446, 1), 'walnut': (654, 1), 'peanut': (567, 1), 'egg': (155, 1), 'apple': (52, 0), 'date': (282, 1), 'apricot': (48, 0), 'avocado': (160, 1), 'banana': (89, 0), 'stawberry': (33, 0), 'cherry': (50, 0), 'blueberry': (57, 0), 'raspberry': (53, 0), 'mango': (60, 0), 'olives': (115, 0), 'peach': (39, 0), 'lemon': (29, 0), 'pear': (57, 0), 'fig': (74, 0), 'pineapple': (50, 0), 'grape': (67, 0), 'kiwi': (61, 0), 'melon': (34, 0), 'orange': (47, 0), 'watermelon': (30, 0), 'steak': (271, 2), 'pork': (242, 2), 'chicken duck': (239, 2), 'sausage': (301, 2), 'fried meat': (301, 2), 'lamb': (294, 2), 'sauce': (68, 0), 'crab': (97, 1), 'fish': (206, 1), 'shellfish': (99, 1), 'shrimp': (99, 1), 'soup': (32, 1), 'bread': (265, 0), 'corn': (86, 0), 'hamburg': (295, 2), 'pizza': (266, 2), 'hanaki baozi': (147, 1), 'wonton dumplings': (284, 1), 'pasta': (131, 1), 'noodles': (138, 1), 'rice': (130, 2), 'pie': (237, 1), 'tofu': (76, 0), 'eggplant': (25, 0), 'potato': (75, 1), 'garlic': (149, 0), 'cauliflower': (25, 0), 'tomato': (19, 0), 'kelp': (43, 0), 'seaweed': (306, 0), 'spring onion': (32, 0), 'rape': (28, 0), 'ginger': (80, 0), 'okra': (33, 0), 'lettuce': (15, 0), 'pumpkin': (26, 0), 'cucumber': (15, 0), 'white radish': (16, 0), 'carrot': (41, 0), 'asparagus': (20, 0), 'bamboo shoots': (27, 0), 'broccoli': (34, 0), 'celery stick': (14, 0), 'cilantro mint': (23, 0), 'snow peas': (42, 0), 'cabbage': (25, 0), 'bean sprouts': (30, 0), 'onion': (40, 0), 'pepper': (20, 0), 'green beans': (31, 0), 'french beans': (31, 0), 'king oyster mushroom': (35, 0), 'shiitake': (34, 0), 'enoki mushroom': (37, 0), 'oyster mushroom': (33, 0), 'white button mushroom': (22, 0), 'salad': (17, 0), 'other ingredients': (0, 0)}

#acceptable file formats
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

#base html
@app.route('/')
def index():
    return render_template('index.html')

#Upload file
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

#Base for initiating inference
@app.route('/inference/<filename>')
def inference(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result = inference_image(filepath)
    return render_template('result.html', filename=filename, result=result)


#Main inference/processing function
def inference_image(image_path):
    start = time.time()

    #Resize image to preset size to speed up processing
    im = Image.open(image_path) #Code from: https://stackoverflow.com/questions/9174338/programmatically-change-image-resolution
    im_resized = im.resize(size)
    im_resized.save(image_path, "JPEG")

    #model for food detection for data processing
    print("inference_detector: ", start)
    model1 = init_detector(config_file, checkpoint_file)
    pred = inference_detector(model1, image_path)

    #model for food detection for image output
    print("DetInfer: ", time.time())
    inferencer = DetInferencer(model=config_file, weights=checkpoint_file, palette=None)
    result = inferencer(image_path, out_dir='static/out/', show=False)

    #to get relevant data into list and dict for processing
    print("pred_ti_dict(): ", time.time() - start)
    pred_masks, pred_dict = pred_to_dict(pred, start)

    #output list init
    arealist = []

    #calculating contours and pixel areas, adding pixel area, pred score, category number, and catehory name to output list
    print("areaList: ", time.time() - start)
    for i in range(len(pred_masks)):
        contours, _ = cv2.findContours(pred_masks[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        temp = [cv2.contourArea(contours[0]), pred_dict['scores'][i], pred_dict['labels'][i] + 1, categories[pred_dict['labels'][i]]]
        arealist.append(temp)

    #getting ratio of pixels to cm^2
    print("ratio: ", time.time() - start)
    ratio = pixel_to_area(image_path, start)

    '''Using ratio to determine real life size of masks, then calculating the
    number of calories in each food item based off its density score: 0 is least dense, 2 is most '''
    print("ratio use: ", time.time() - start)
    for i in arealist:
        tempArea = i[0] * ratio
        if cal_per_100g.get(categories[(i[2]-1)])[1] == 0:
            tempCalories = tempArea / 200
            tempCalories = tempCalories * cal_per_100g.get(categories[(i[2]-1)])[0]
        elif cal_per_100g.get(categories[(i[2]-1)])[1] == 1:
            tempCalories = tempArea / 150
            tempCalories = tempCalories * cal_per_100g.get(categories[(i[2]-1)])[0]
        elif cal_per_100g.get(categories[(i[2]-1)])[1] == 2:
            tempCalories = tempArea / 100
            tempCalories = tempCalories * cal_per_100g.get(categories[(i[2]-1)])[0]

        i.append(tempArea)
        i.append(tempCalories)
        print(i)


    print("return: ", time.time() - start)
    return arealist


#turning predictions into dictionary, as well as cinverting output onto suitable format for area finding
def pred_to_dict(pred, start):
    # Code form sizing_module.py provided by Chris Moorhead, modfied for numpy output
    print("pred to dict start: ", time.time() - start)
    pred_masks = pred.pred_instances.masks.clone().detach().cpu().numpy()
    pred_scores = pred.pred_instances.scores.clone().detach().cpu().numpy()
    pred_labels = pred.pred_instances.labels.clone().detach().cpu().numpy()
    pred_dict = {"masks": pred_masks, "scores": pred_scores, "labels": pred_labels}

    #Converting array of tensors to numpy array of ints for contour/area fining
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

#finding ratio of pixel to cm^2
def pixel_to_area(img_path, start):
    #initalising sticker detection model, converting output to dictionary/list
    print("pixeltoarea start: ", time.time() - start)
    model2 = init_detector(sticker_config, sticker_checkpoint)
    pred_sticker = inference_detector(model2, img_path)
    pred_mask_sticker, pred_dict_sticker = pred_to_dict(pred_sticker, start)

    #calc stickers real life area based on user input diamter
    sticker_area = (DIAMETER / 2)**2 * pi
    pixel_to_area_ratio = 0
    arealistSticker = []

    #generating the masks of the sticker along with pixel area of sticker mask
    print("pixeltoarea contour start: ", time.time() - start)
    for i in range(len(pred_mask_sticker)):
        contours, _ = cv2.findContours(pred_mask_sticker[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        temp = [cv2.contourArea(contours[0]), pred_dict_sticker['scores'][i], pred_dict_sticker['labels'][i]] # categories[int(pred_dict_sticker['labels'][i])]
        arealistSticker.append(temp)

    #taking most confident prediction of sticker
    max_index = arealistSticker[1].index(max(arealistSticker[1]))

    #calculating ratio if class of prediciton is sticker (which it should be 100% as no other classes exist in this model)
    if arealistSticker[max_index][2] == 0:
        pixel_to_area_ratio = sticker_area / arealistSticker[0][max_index]

    print("pixeltoarea return: ", time.time() - start)
    return pixel_to_area_ratio


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
