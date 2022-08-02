# import dependencies 
import splitfolders 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from keras import models, layers
import tensorflow as tf
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback


# building model function
def buildModel(input_shape, output_shape):
    
    model = models.Sequential()
    model.add(layers.Conv2D(8, (3,3), activation= 'relu', input_shape= (288,432,4), padding= 'same'))
    model.add(layers.MaxPooling2D((4,4), padding= 'same'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(16, (3,3), activation= 'relu', padding= 'same'))
    model.add(layers.MaxPooling2D((4,4), padding= 'same'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(32, (3,3), activation= 'relu', padding= 'same'))
    model.add(layers.MaxPooling2D((4,4), padding= 'same'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
    model.add(layers.MaxPooling2D((4,4), padding= 'same'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
    model.add(layers.MaxPooling2D((4,4), padding= 'same'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation= 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation= 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation= 'softmax'))
    
    return model


# training function
def trainModel(model):
    run = wandb.init(project = 'spoken-digit-classification',
                     config = {  
                         "epochs": 100,
                         "loss_function": "categorical_crossentropy",
                         "architecture": "CNN",
                         "dataset": "free-spoken-digit"
                     })
    
    # use this to configure our experiment
    config = run.config

    # initialize model
    tf.keras.backend.clear_session()
    
    # compile model
    optimizer = tf.keras.optimizers.Adam() 
    model.compile(optimizer, config.loss_function, metrics = ['accuracy'])
    
    # model fit
    return model.fit(train_generator, epochs = config.epochs, validation_data=val_generator, callbacks = [WandbCallback()])


# evaluation function
def evaluateModel(model):
    
    loss, acc = model.evaluate(val_generator)
    
    return loss, acc


# main
if __name__ == '__main__':
    
    # status: import python librairies
    print("Import dependencies and wandb login...\n")
    
    # /!\ /!\ /!\ /!\ /!\ /!\
    # Replace MY_WANDB_API_KEY by your wandb API key
    os.environ["WANDB_API_KEY"] = "MY_WANDB_API_KEY"
    # /!\ /!\ /!\ /!\ /!\ /!\
    
    classes_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    
    # status: start of data preprocessing
    print("Data is being preprocessed...\n")
    
    # use keras data generator for image classification
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_dir = "/workspace/data/spectrograms_split/train/"
    train_generator = datagen.flow_from_directory(train_dir, target_size=(288,432), color_mode="rgba", class_mode='categorical', classes=classes_names, batch_size=128)
    
    val_dir = "/workspace/data/spectrograms_split/val/"
    val_generator = datagen.flow_from_directory(val_dir, target_size=(288,432), color_mode='rgba', class_mode='categorical', classes=classes_names, batch_size=128)
    
    # status: end of data preprocessing
    print("Data preprocessing is finished!\n")
        
    # define input and output sizes
    img_shape = (288,432,4)
    nmb_classes = 10
    
    # build the model
    model = buildModel(img_shape, nmb_classes)
    
    # status: start of model training
    print("Model training:")
    
    # launch the training
    model_history = trainModel(model)
    
    # evaluate the model
    print("\nModel evaluation:")
    test_loss, test_acc = evaluateModel(model)
    
    # with wandb.log you can pass in metrics as key-value pairs
    run.log({'Test error rate': round((1 - accuracy) * 100, 2)})

    # stop run
    run.finish()
