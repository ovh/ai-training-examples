import numpy as np
import pandas as pd

# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# model
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential

df = pd.read_csv('/workspace/data/data.csv')
# dataframe shape
df.shape
# dataframe types
df.dtypes

class_list = df.iloc[:,-1]
encoder = LabelEncoder()
y = encoder.fit_transform(class_list)
print("y: ", y)

input_parameters = df.iloc[:, 1:27]
scaler = StandardScaler()
X = scaler.fit_transform(np.array(input_parameters))
print("X:", X)

# training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

model = load_model('/workspace/saved_model/my_model')
print(model.summary())
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
batch_size = 128
model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 100, batch_size = batch_size)
model.save('/workspace/saved_model/my_model2')
print('End of training')
