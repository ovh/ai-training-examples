from ultralytics import YOLO
import streamlit as st

#######################################################################################################################
## 🎯 The aim of this script is to create an Rock/Paper/Scissors application based on a trained model (from YOLOv8). ##
## 🏞 The uploaded snapshots are stored in /workspace/                                                               ##
## 🧠 The train model is stored in /workspace/model/rock-paper-scissors/                                             ##
#######################################################################################################################

# Save uploaded photo
def save_photo(photo):
    
    photoAbsolutePath = '/workspace/' + photo.name
    
    with open(photoAbsolutePath,'wb') as f:
         f.write(photo.getbuffer())
    
    return photoAbsolutePath

# main
if __name__ == '__main__':

    st.write("## Welcome on the 🪨 📄 ✂️ game!")
    # 🧠 Load the model
    model = YOLO('/workspace/model/rock-paper-scissors/best.torchscript')

    # 📸 Camera input
    img_file_buffer = st.camera_input("Take your picture in real time:")
    if img_file_buffer is not None:
      photoPath = save_photo(img_file_buffer) 

      # 🔎 Prediction
      results = model.predict(photoPath, verbose=True, save=True, conf=0.5)

      # 📈 Display results
      for r in results:
        for c in r.boxes.cls:
          st.write(r.names[int(c)])