from flask import Flask, render_template, request, redirect, url_for, jsonify
from mmdet.apis import init_detector, inference_detector, DetInferencer
import os
import time
import cv2
from pycocotools import mask
import numpy
from mmdet.registry import VISUALIZERS
from math import pi

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Initialize your mmdetection model
config_file = 'C:/Users/peter/mmdetection/configs/AI4Food/configs/foodseg103coco.py'
checkpoint_file = 'C:/Users/peter/mmdetection/work_dirs/foodseg103coco/epoch_25.pth'
sticker_config = 'C:/Users/peter/mmdetection/configs/AI4Food/configs/stickerPre.py'
sticker_checkpoint = 'C:/Users/peter/mmdetection/work_dirs/stickerPre/epoch_4.pth'
# model = init_detector(config_file, checkpoint_file)
DIAMETER = 0.9

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
    # Initialize the DetInferencer
    model1 = init_detector(config_file, checkpoint_file)
    pred = inference_detector(model1, image_path)

    inferencer = DetInferencer(model=config_file, weights=checkpoint_file, palette=None)
    result = inferencer(image_path, out_dir='static/out/', show=False)

    pred_masks, pred_dict = pred_to_dict(pred)

    arealist = []

    for i in range(len(pred_masks)):
        contours, _ = cv2.findContours(pred_masks[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        temp = [cv2.contourArea(contours[0]), pred_dict['scores'][i], pred_dict['labels'][i] + 1]
        arealist.append(temp)

    ratio = pixel_to_area(image_path)

    for i in arealist:
        tempArea = i[0] * ratio
        i.append(tempArea)
        print(i)

    return arealist

# Code form sizing_module.py provided by Chris Moorhead, modfied for numpy output
def pred_to_dict(pred):
    pred_masks = pred.pred_instances.masks.clone().detach()
    pred_scores = pred.pred_instances.scores.clone().detach()
    pred_labels = pred.pred_instances.labels.clone().detach()
    pred_dict = {"masks": pred_masks.cpu().numpy(), "scores": pred_scores.cpu().numpy(), "labels": pred_labels.cpu().numpy()}

    pred_masks = []
    for i in pred_dict['masks']:
        temp = []
        for j in i:
            temp1 = []
            for x in j:
                xint = int(x)
                temp1.append(xint)

            if len(temp1) != 0:
                temp1 = numpy.array(temp1, dtype=numpy.uint8)
                temp.append(temp1)

        if len(temp) != 0:
            temp = numpy.array(temp)
            pred_masks.append(temp)

    pred_masks = numpy.array(pred_masks)

    return pred_masks, pred_dict

def pixel_to_area(img_path):
    model2 = init_detector(sticker_config, sticker_checkpoint)
    pred_sticker = inference_detector(model2, img_path)
    pred_mask_sticker, pred_dict_sticker = pred_to_dict(pred_sticker)
    sticker_area = (DIAMETER / 2)**2 * pi
    pixel_to_area_ratio = 0
    arealistSticker = []

    for i in range(len(pred_mask_sticker)):
        contours, _ = cv2.findContours(pred_mask_sticker[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        temp = [cv2.contourArea(contours[0]), pred_dict_sticker['scores'][i], pred_dict_sticker['labels'][i]]
        arealistSticker.append(temp)

    max_index = arealistSticker[1].index(max(arealistSticker[1]))

    if arealistSticker[max_index][2] == 0:
        pixel_to_area_ratio = sticker_area / arealistSticker[0][max_index]

    return pixel_to_area_ratio

def area_to_cal(maskList):
    pass

if __name__ == '__main__':
    app.run(debug=True)
