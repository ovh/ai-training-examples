import io
import os
import shutil
import json
from PIL import Image

import base64
import cv2
import torch
import flask
from flask import Flask, jsonify, url_for, render_template, request, redirect, make_response
import numpy as np

app = Flask(__name__)

model_yolov5s = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5s1.pt', force_reload=True)  # default
model_yolov5m = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5m1.pt', force_reload=True)  # default

<<<<<<< HEAD

=======
model_yolov5s = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape() 
model_yolov5s.eval()

model_yolov5m = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True).autoshape() 
model_yolov5m.eval()

# yolov5s
>>>>>>> a0cd5822b30bae7eac324dad981e37ed72dea2df
def get_prediction_yolov5s(img_bytes):
    #img = Image.open(io.BytesIO(img_bytes))
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    imgs = [img]  # batched list of images
    # Inference
<<<<<<< HEAD
    results = model_yolov5s(imgs, size=640)  
=======
    results = model_yolov5s(imgs, size=640) 
>>>>>>> a0cd5822b30bae7eac324dad981e37ed72dea2df
    return results

# yolov5m
def get_prediction_yolov5m(img_bytes):
    #img = Image.open(io.BytesIO(img_bytes))
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    imgs = [img]  # batched list of images
    # Inference
    results = model_yolov5m(imgs, size=640) 
    return results


@app.route('/', methods=['GET'])
def get():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
<<<<<<< HEAD
    print(f'User selected model : {request.form.get("model_choice")}')

    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files.get('file')
    if not file:
        return
    
    img_bytes = file.read()
    if request.form.get("model_choice") == 'yolov5s':
        results = get_prediction_yolov5s(img_bytes)
    if request.form.get("model_choice") == 'yolov5m':
        results = get_prediction_yolov5m(img_bytes)
    
    results.render()  # updates results.imgs with boxes and labels
    
    for img in results.imgs:
        im_arr = cv2.imencode('.jpg', img)[1]
        response = make_response(im_arr.tobytes())
        response.headers['Content-Type'] = 'image/jpeg'
    return response
=======
    if request.method == 'POST':
        # model choice
        print(f'User selected model : {request.form.get("model_choice")}')

        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        
        img_bytes = file.read()
        if request.form.get("model_choice") == 'yolov5s':
            results = get_prediction_yolov5s(img_bytes)
        if request.form.get("model_choice") == 'yolov5m':
            results = get_prediction_yolov5m(img_bytes)
        results.save("static/")  # save the results
>>>>>>> a0cd5822b30bae7eac324dad981e37ed72dea2df


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
