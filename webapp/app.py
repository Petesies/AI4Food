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

    image_mask_np_uint8 = numpy.uint8(pred.pred_instances['masks'].cpu())  # Ensure uint8 format
    image_mask_np_uint8 = numpy.squeeze(image_mask_np_uint8)

    # Find contours in the mask
    contours, _ = cv2.findContours(image_mask_np_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming you want to work with the largest contour, you can sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # contours = cv2.findContours(pred.pred_instances)
    print(cv2.contourArea(contours))
    # print(cv2.contourArea(pred.pred_instances['masks'].cpu().numpy()))

    # import struct
    # b = result['predictions'][0]['masks'][1]['counts'].encode('utf-8')

    # value = struct.unpack('B', b)[0]

    # print("VALUE: ", b)

    print("Result Dict Keys: ", list(result.keys()))
    print("Predictions Dict Keys: ", list(result['predictions'][0].keys()))
    print("Labels: ", result['predictions'][0]['labels'])
    print("Scores: ", result['predictions'][0]['scores'])
    print("bboxes: ", result['predictions'][0]['bboxes'])
    print("masks: ", result['predictions'][0]['masks'][1])
    print("masks keys: ", list(result['predictions'][0]['masks'][1].keys()))
    print("Counts type: ", type(result['predictions'][0]['masks'][0]['counts']))
    # print("Visualization list single list: ", [v for v in result['visualization'][0][5]])
    print(len(result['predictions'][0]['masks']))
    # print([v for v in mask.decode(result['predictions'][0]['masks'][5])])
    # print("area: ", cv2.contourArea(mask.decode(result['predictions'][0]['masks'][5])))


    return result

if __name__ == '__main__':
    app.run(debug=True)
