import ultralytics
from ultralytics import YOLO
import shutil

#######################################################################################################################
## 🎯 The aim of this script is to do transfert learning on YOLOv8 model.                                            ##
## 💿 The data for train the model are in /workspace/data/rock-paper-scissors/                                       ##
## 🧠 The train model are stored in /workspace/model/rock-paper-scissors/                                            ##
#######################################################################################################################

# ✅ Check configuration
ultralytics.checks()

# 🧠 Load a pretrained YOLO model
model = YOLO('yolov8n.pt')

# 💪 Train the model with new data ➡️ one GPU / 50 iterations (epochs)
model.train(data='/workspace/data/rock-paper-scissors/data.yaml', device=0, epochs=50, verbose=True)

# 💾 Save the model
exportedMetaData = model.export()
print('Model save to : ' + exportedMetaData)

# ➡️ Copy the model to the object storage
shutil.copy(exportedMetaData, '/workspace/model/rock-paper-scissors/')