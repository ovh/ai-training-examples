# import dependencies 
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models, layers
import wandb
from wandb.keras import WandbCallback


# processing function
def processData(df):
    
    # input parameters
    class_list = df.iloc[:,-1]
    encoder = LabelEncoder()
    output = encoder.fit_transform(class_list)

    # output - classification result
    input_parameters = df.iloc[:, 1:27]
    scaler = StandardScaler()
    inputs = scaler.fit_transform(np.array(input_parameters))
    
    return inputs, output


# building function
def buildModel(input_shape, output_shape):
    
    model = models.Sequential()
    model.add(layers.Dense(512, activation = 'relu', input_shape = (input_shape,)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output_shape, activation = 'softmax'))

    return model


# training function
def trainModel(model, run):

    # use this to configure our experiment
    config = run.config

    # initialize model
    tf.keras.backend.clear_session()
    
    # compile model
    optimizer = tf.keras.optimizers.Adam() 
    model.compile(optimizer, config.loss_function, metrics = ['accuracy'])
    
    return model.fit(X_train, y_train, epochs = config.epochs, validation_data = (X_val, y_val), batch_size = config.batch_size, callbacks = [WandbCallback()])


# evaluation function
def evaluateModel(model):
    
    loss, acc = model.evaluate(X_val, y_val)
    
    return loss, acc


# main
if __name__ == '__main__':
    
    # status: import python librairies
    print("Import dependencies and wandb login...\n")
   
    # /!\ /!\ /!\ /!\ /!\ /!\
    # Replace MY_WANDB_API_KEY by your wandb API key
    os.environ["WANDB_API_KEY"] = "MY_WANDB_API_KEY"
    # /!\ /!\ /!\ /!\ /!\ /!\
    
    # create the dataframe with the csv file
    dataframe = pd.read_csv('/workspace/data/csv_files/data_3_sec.csv')
    
    # status: start of data preprocessing
    print("Data is being preprocessed...\n")
    X, y = processData(dataframe)
    
    # training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
    
    # status: end of data preprocessing
    print("Data preprocessing is finished!\n")
        
    # define input and output sizes
    nmb_parameters = X_train.shape[1]
    nmb_classes = 10
    
    # build the model
    model = buildModel(nmb_parameters, nmb_classes)
    
    # wandb init
    run = wandb.init(project = 'spoken-digit-classification',
                     config = {  
                         "epochs": 100,
                         "loss_function": "sparse_categorical_crossentropy",
                         "batch_size": 128,
                         "architecture": "ANN",
                         "dataset": "free-spoken-digit"
                     })
    
    # status: start of model training
    print("Model training:")
    
    # launch the training
    model_history = trainModel(model, run)
    
    # evaluate the model
    print("\nModel evaluation:")
    test_loss, test_acc = evaluateModel(model)

    # stop run
    run.finish()
