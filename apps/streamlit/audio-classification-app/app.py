# import dependences
import streamlit as st
import librosa
import csv
import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder



# save sound file, uploaded before, in a folder
def save_file(sound_file):
    
    # save your sound file in the right folder by following the path 
    with open(os.path.join('audio_files/', sound_file.name),'wb') as f:
         f.write(sound_file.getbuffer())
    
    return sound_file.name



# transform the sound into a csv file
def transform_wav_to_csv(sound_saved):
    
    # define the column names
    header_test = 'filename length chroma_stft_mean chroma_stft_var rms_mean rms_var spectral_centroid_mean spectral_centroid_var spectral_bandwidth_mean \
        spectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var harmony_mean harmony_var perceptr_mean perceptr_var tempo mfcc1_mean mfcc1_var mfcc2_mean \
        mfcc2_var mfcc3_mean mfcc3_var mfcc4_mean mfcc4_var'.split()
    
    # create the csv file
    file = open(f'csv_files/{os.path.splitext(sound_saved)[0]}.csv', 'w', newline = '')
    with file:
        writer = csv.writer(file)
        writer.writerow(header_test)
        
    # calculate the value of the librosa parameters
    sound_name = f'audio_files/{sound_saved}'
    y, sr = librosa.load(sound_name, mono = True, duration = 30)
    chroma_stft = librosa.feature.chroma_stft(y = y, sr = sr)
    rmse = librosa.feature.rms(y = y)
    spec_cent = librosa.feature.spectral_centroid(y = y, sr = sr)
    spec_bw = librosa.feature.spectral_bandwidth(y = y, sr = sr)
    rolloff = librosa.feature.spectral_rolloff(y = y, sr = sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y = y, sr = sr)
    to_append = f'{os.path.basename(sound_name)} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    
    # fill in the csv file
    file = open(f'csv_files/{os.path.splitext(sound_saved)[0]}.csv', 'a', newline = '')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
    
    # create test dataframe
    df_test = pd.read_csv(f'csv_files/{os.path.splitext(sound_saved)[0]}.csv')
    
    # each time you add a sound, a line is added to the test.csv file
    # if you want to display the whole dataframe, you can deselect the following line
    #st.write(df_test)
    
    return df_test



# classify the uploded sounds
def classification(dataframe):
    
    # create a dataframe with the csv file of the data used for training and validation
    df = pd.read_csv('csv_files/data.csv')
    
    # OUTPUT: labels => last column
    labels_list = df.iloc[:,-1]
    
    # encode the labels (0 => 44)
    converter = LabelEncoder()
    y = converter.fit_transform(labels_list)
    
    # INPUTS: all other columns are inputs except the filename
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(df.iloc[:, 1:27]))
    X_test = scaler.transform(np.array(dataframe.iloc[:, 1:27]))

    # load the pretrained model
    model = load_model('saved_model/my_model2')
    
    # generate predictions for test samples
    predictions = model.predict(X_test)
    
    # generate argmax for predictions
    classes = np.argmax(predictions, axis = 1)
    
    # transform class number into class name
    result = converter.inverse_transform(classes)

    return result



# if you have chosen prediction in the sidebar
def choice_prediction():
    st.write('# Prediction')
    st.write('### Choose a marine mammal sound file in .wav format')
    
    # upload sound
    uploaded_file = st.file_uploader(' ', type='wav')
    
    if uploaded_file is not None:
            
        # view details
        file_details = {'filename':uploaded_file.name, 'filetype':uploaded_file.type, 'filesize':uploaded_file.size}
        st.write(file_details)
        
        # read and play the audio file
        st.write('### Play audio')
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')
      
        # save_file function
        save_file(uploaded_file)

        # define the filename
        sound = uploaded_file.name
        
        # transform_wav_to_csv function
        transform_wav_to_csv(sound)
        
        st.write('### Classification results')
        
        # if you select the predict button
        if st.button('Predict'):
            # write the prediction: the prediction of the last sound sent corresponds to the first column
            st.write("The marine mammal is: ",  str(classification(transform_wav_to_csv(sound))).replace('[', '').replace(']', '').replace("'", '').replace('"', ''))

    else:
        st.write('The file has not been uploaded yet')
    
    return
        
        
        
# main
if __name__ == '__main__':
    
    st.image(Image.open('logo_ovh.png'), width=200)
    st.write('___')
    
    # create a sidebar
    st.sidebar.title('Marine mammal sounds classification')
    select = st.sidebar.selectbox('', ['Marine mammals', 'Prediction'], key='1')
    st.sidebar.write(select)
    
    # if sidebar selection is "Prediction"
    if select=='Prediction':
        # choice_prediction function
        choice_prediction()
    
    # else: stay on the home page
    else:
        st.write('# Marine mammals')
        st.write('The different marine mammals studied are the following.')
        st.write('For more information, please refer to this [link](https://cis.whoi.edu/science/B/whalesounds/index.cfm).')
        st.image(Image.open('marine_mammal_animals.png'))
