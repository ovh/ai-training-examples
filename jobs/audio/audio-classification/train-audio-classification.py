import numpy as np
import pandas as pd
import datetime

# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# model
import tensorflow as tf

########################################################################################################################################################
# The goal of this script is to train a pre-construct model to recognize marine mammal sound.                                                          #
# See the Notebook "notebook-marine-sound-classification" in the ai-training-examples for                                                              #
# more details : https://github.com/ovh/ai-training-examples/blob/main/notebooks/audio/audio-classification/notebook-marine-sound-classification.ipynb #
# You must mount 2 volumes for the data and the model (the same used for the Notebook for example 😉) :                                                #
#   - /workspace/saved_model where the model is stored                                                                                                 #
#   - /workspace/data/csv/ where store the data for the training                                                                                       #
########################################################################################################################################################


# 🗃 Load pre-transform data
df = pd.read_csv('/workspace/data/data.csv')
# dataframe shape
df.shape
# dataframe types
df.dtypes

# 🔢 Encode the labels (0 => 44) 
class_list = df.iloc[:,-1]
encoder = LabelEncoder()
y = encoder.fit_transform(class_list)
print("y: ", y)

# 🧹 Uniformize data thanks to the initial data 
input_parameters = df.iloc[:, 1:27]
scaler = StandardScaler()
X = scaler.fit_transform(np.array(input_parameters))
print("X:", X)

# ⚗️ Create training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

# 🧠 Define model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(45, activation='softmax'),
])

print(model.summary())

# 💪 Train the model with data
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')

# 📈 Add the TensorBoard callback (optional)
print('Model tracking')
log_dir = "/workspace/saved_model/runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 100, batch_size = 128, callbacks = [tensorboard_callback])

# 💿 Save the model for future usages
model.save('/workspace/saved_model/my_model2')
print('End of training')
