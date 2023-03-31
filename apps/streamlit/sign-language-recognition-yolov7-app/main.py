# import dependences
import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2
import io
import os

# load the model and put it in cache
@st.cache
def load_model():

    # load the custom model for asl recognition
    custom_yolov7_model = torch.hub.load("WongKinYiu/yolov7", 'custom', '/workspace/asl-volov7-model/yolov7.pt')

    return custom_yolov7_model


# inference function
def get_prediction(img_bytes, model):

    img = Image.open(io.BytesIO(img_bytes))
    # inference
    results = model(img, size=640)

    return results


# image analysis function
def analyse_image(image, model):

    if image is not None:

        # load image
        img = Image.open(image)

        # detect sign on image
        bytes_data = image.getvalue()
        img_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        result = get_prediction(img_bytes, model)
        result.render()

        for img in result.imgs:
            RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_arr = cv2.imencode('.jpg', RGB_img)[1]
            st.image(im_arr.tobytes())

        result_list = list((result.pandas().xyxy[0])["name"])

    else:
        st.write("no asl letters were detected!")
        result_list = []

    # return detected letters in a list
    return result_list


# create the word
def display_letters(letters_list):

    word = ''.join(letters_list)
    path_file = "/workspace/word_file.txt"
    with open(path_file, "a") as f:
        f.write(word)

    return path_file


# main
if __name__ == '__main__':

    st.image("/workspace/head-asl-yolov7-app.png")
    st.write("## Welcome on your ASL letters recognition app!")

    # import yolov7 model for asl recognition
    model = load_model()

    # camera input
    img_file_buffer = st.camera_input("Take your picture in real time:")

    # result os detected ASL letters
    result_list = analyse_image(img_file_buffer, model)
    path_file = display_letters(result_list)

    # clear the resulting word
    if st.button("Clear result"):
        if os.path.isfile(path_file):
            os.remove(path_file)
            print("File has been deleted")
        else:
            print("File does not exist")

    # display the result if it exists
    if (os.path.exists(path_file)==True):
        with open(path_file, "r") as f:
            content = f.read()
            st.write(content)
            f.close()
    else:
        pass
