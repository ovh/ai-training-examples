import io
import os
import shutil
import json
from PIL import Image
import cv2
import torch
import flask
from flask import Flask, jsonify, url_for, render_template, request, redirect

app = Flask(__name__)

# the files yolov5s1.pt and yolov5m1.pt are located in the /models folder
model_yolov5s = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5s1.pt')  # default
model_yolov5m = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5m1.pt')  # default

# yolov5s
def get_prediction_yolov5s(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images
    # inference
    results = model_yolov5s(imgs, size=640) 
    return results

# yolov5m
def get_prediction_yolov5m(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images
    # inference
    results = model_yolov5m(imgs, size=640) 
    return results

# get method
@app.route('/', methods=['GET'])
def get():
    return render_template('index.html')

# post method
@app.route('/', methods=['POST'])
def predict():
    print(f'User selected model : {request.form.get("model_choice")}')

    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files.get('file')
    if not file:
        return
    
    img_bytes = file.read()
    
    # choice of the model
    if request.form.get("model_choice") == 'yolov5s':
        results = get_prediction_yolov5s(img_bytes)
    if request.form.get("model_choice") == 'yolov5m':
        results = get_prediction_yolov5m(img_bytes)
    
    # updates results.imgs with boxes and labels
    results.render()  
    
    # encode the resulting image and return it
    for img in results.imgs:
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_arr = cv2.imencode('.jpg', img)[1]
        response = make_response(im_arr.tobytes())
        response.headers['Content-Type'] = 'image/jpeg'
    return response

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
