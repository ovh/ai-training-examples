# import dependencies
import gradio as gr
import tensorflow as tf
import cv2

# app title
title = "Welcome on your first sketch recognition app!"

# app description
head = (
  "<center>"
  "<img src='file/mnist-classes.png' width=400>"
  "The robot was trained to classify numbers (from 0 to 9). To test it, write your number in the space provided."
  "</center>"
)

# GitHub repository link
ref = "Find the whole code [here](https://github.com/ovh/ai-training-examples/tree/main/apps/gradio/sketch-recognition)."

# image size: 28x28
img_size = 28

# classes name (from 0 to 9)
labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# load model (trained on MNIST dataset)
model = tf.keras.models.load_model("model/sketch_recognition_numbers_model.h5")

# prediction function for sketch recognition
def predict(img):

  # image shape: 28x28x1
  img = cv2.resize(img, (img_size, img_size))
  img = img.reshape(1, img_size, img_size, 1)

  # model predictions
  preds = model.predict(img)[0]

  # return the probability for each classe
  return {label: float(pred) for label, pred in zip(labels, preds)}

# top 3 of classes
label = gr.outputs.Label(num_top_classes=3)

# open Gradio interface for sketch recognition
interface = gr.Interface(fn=predict, inputs="sketchpad", outputs=label, title=title, description=head, article=ref)
interface.launch(server_name="0.0.0.0", server_port=8080)
