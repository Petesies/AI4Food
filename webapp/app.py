from flask import Flask, render_template, request, redirect, url_for, jsonify
from mmdet.apis import init_detector, inference_detector, DetInferencer
import os
import time
import cv2
from pycocotools import mask
import numpy

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Initialize your mmdetection model
config_file = 'C:/Users/peter/mmdetection/configs/AI4Food/configs/foodseg103coco.py'
checkpoint_file = 'C:/Users/peter/mmdetection/work_dirs/foodseg103coco/epoch_25.pth'
model = init_detector(config_file, checkpoint_file)

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
    pred = inference_detector(model, image_path)
    inferencer = DetInferencer(model=config_file, weights=checkpoint_file, palette=None)

    # Perform inference
    result = inferencer(image_path, out_dir='static/out/', show=False)

    pred_dict = pred_to_dict(pred)

    # pred_masks = numpy.array([])
    pred_masks = []
    for i in pred_dict['masks']:
        # temp = numpy.array([])
        temp = []
        for j in i:
            # temp1 = numpy.array([])
            temp1 = []
            for x in j:
                # if int(x) != 0:
                    # print(int(x))
                xint = int(x)
                temp1.append(xint)
                # numpy.append(temp1, xint)
                # print(temp1)

            # numpy.append(temp, temp1)
            if len(temp1) != 0:
                temp1 = numpy.array(temp1, dtype=numpy.uint8)
                temp.append(temp1)
        # numpy.append(pred_masks, temp)
        if len(temp) != 0:
            temp = numpy.array(temp)
            pred_masks.append(temp)

    pred_masks = numpy.array(pred_masks)
    print(type(pred_masks[0][0][0]))


    # print(type(int(pred_dict['masks'][0][0])))
    # print(pred_masks)
    print("Popo")


    print(len(pred_dict['scores']))
    print(len(pred_dict['labels']))
    for i in range(len(pred_masks)):
        contours, _ = cv2.findContours(pred_masks[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print(cv2.contourArea(contours[0]))
        print(pred_dict['scores'][i])
        print(pred_dict['labels'][i])
    # print("VALUE: ", b)
    # model1.show_result(image_path, pred)
    # print("Result Dict Keys: ", list(result.keys()))
    # print("Predictions Dict Keys: ", list(result['predictions'][0].keys()))
    # print("Labels: ", result['predictions'][0]['labels'])
    # print("Scores: ", result['predictions'][0]['scores'])
    # print("bboxes: ", result['predictions'][0]['bboxes'])
    # print("masks: ", result['predictions'][0]['masks'][1])
    # print("masks keys: ", list(result['predictions'][0]['masks'][1].keys()))
    # print("Counts type: ", type(result['predictions'][0]['masks'][0]['counts']))
    # # print("Visualization list single list: ", [v for v in result['visualization'][0][5]])
    # print(len(result['predictions'][0]['masks']))
    # print([v for v in mask.decode(result['predictions'][0]['masks'][5])])
    # print("area: ", cv2.contourArea(mask.decode(result['predictions'][0]['masks'][5])))


    return result

def pred_to_dict(pred):
    pred_masks = pred.pred_instances.masks.clone().detach()
    pred_scores = pred.pred_instances.scores.clone().detach()
    pred_labels = pred.pred_instances.labels.clone().detach()
    return {"masks": pred_masks.cpu().numpy(), "scores": pred_scores.cpu(), "labels": pred_labels.cpu()}

if __name__ == '__main__':
    app.run(debug=True)
