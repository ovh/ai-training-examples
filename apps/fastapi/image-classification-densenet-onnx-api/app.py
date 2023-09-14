# import dependencies
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from torchvision import transforms
import onnxruntime
from PIL import Image
from io import BytesIO
import itertools

# initialize an instance of fastapi
app = FastAPI()

# load the ONNX model
session = onnxruntime.InferenceSession("/workspace/models/densenet_onnx_cifar10/1/densenet_onnx_cifar10.onnx", device="cuda")

# dictionary with class name and index
idx_to_class = {0: 'AIRPLANE', 1: 'AUTOMOBILE', 2: 'BIRD', 3: 'CAT', 4: 'DEER', \
                5: 'DOG', 6: 'FROG', 7: 'HORSE', 8: 'SHIP', 9: 'TRUCK'}


# /// PROCESS IMAGE \\\
def process_img(file) -> Image.Image:
    
    # transform the image
    transform = transforms.Compose([
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    # load image
    test_image = Image.open(BytesIO(file))

    # apply the transformations to the input image and convert it into a tensor
    test_image_tensor = transform(test_image).unsqueeze(0)

    # make the input image ready to be input as a batch of size 1
    test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    # convert the tensor to numpy array
    np_image = test_image_tensor.numpy()
    
    return np_image.astype(np.float32)

    
# /// GET PREDICTION \\\
def get_prediction(data):
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
 
    # get prediction from model
    result = session.run([output_name], {input_name: data})
    
    return result


# /// GET METHOD \\\
@app.get('/')
def root():
    return {'message': 'Welcome to the Image Classification API'}


# /// POST METHOD \\\
@app.post("/uploadimage/")
async def create_upload_file(file: bytes = File(...)):
    
    # transform
    data = process_img(file)
    
    # prediction
    result = get_prediction(data)
    
    # create a dict with classes and scores
    predictions_result = {}
    for i in range(10):
        p = np.array(result).squeeze()
        predictions_result[idx_to_class[i]]=p[i]
    
    # sort dict by top of predictions
    top_classes = dict(sorted(predictions_result.items(), key=lambda x:x[1], reverse=True))
    for value in top_classes:
        top_classes[value] = top_classes[value].item()
    
    # return top 3
    return dict(itertools.islice(top_classes.items(), 3))
