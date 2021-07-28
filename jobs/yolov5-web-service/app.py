import io
from PIL import Image
import cv2
import torch
from flask import Flask, render_template, request, make_response
from werkzeug.exceptions import BadRequest

# creating flask app
app = Flask(__name__)

# the files yolov5s1.pt and yolov5m1.pt are located in the /models folder
model_yolov5s = torch.hub.load('ultralytics/yolov5', 'custom', path='models_train/yolov5s_100epochs.pt', force_reload=True)  # default
model_yolov5m = torch.hub.load('ultralytics/yolov5', 'custom', path='models_train/yolov5m_100epochs.pt', force_reload=True)  # default

# yolov5s
def get_prediction_yolov5s(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    # inference
    results = model_yolov5s(img, size=640)  
    return results

# yolov5m
def get_prediction_yolov5m(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    # inference
    results = model_yolov5m(img, size=640) 
    return results

# get method
@app.route('/', methods=['GET'])
def get():
    return render_template('index.html')

# post method
@app.route('/', methods=['POST'])
def predict():
    file = extract_img(request)
    img_bytes = file.read()
    
    # choice of the model
    if request.form.get("model_choice") == 'yolov5s':
        results = get_prediction_yolov5s(img_bytes)
    if request.form.get("model_choice") == 'yolov5m':
        results = get_prediction_yolov5m(img_bytes)
    
    print(f'User selected model : {request.form.get("model_choice")}')
    
    # updates results.imgs with boxes and labels
    results.render()
    
    # encoding the resulting image and return it
    for img in results.imgs:
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_arr = cv2.imencode('.jpg', RGB_img)[1]
        response = make_response(im_arr.tobytes())
        response.headers['Content-Type'] = 'image/jpeg'
    return response

def extract_img(request):
    # checking if image uploaded is valid
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")

    file = request.files['file']
    if file.filename == '':
        raise BadRequest("Given file is invalid")
    
    return file
    
if __name__ == '__main__':
    # starting app
    app.run(debug=True,host='0.0.0.0')
