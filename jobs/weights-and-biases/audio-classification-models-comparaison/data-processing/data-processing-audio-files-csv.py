# import dependencies 
import librosa
import csv
import os
import numpy as np
import pandas as pd


# dataframe creation function
def createDataframe():
    
    # columns name
    header = "filename length chroma_stft_mean chroma_stft_var rms_mean rms_var spectral_centroid_mean spectral_centroid_var spectral_bandwidth_mean spectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var harmony_mean harmony_var perceptr_mean perceptr_var tempo mfcc1_mean mfcc1_var mfcc2_mean mfcc2_var mfcc3_mean mfcc3_var mfcc4_mean mfcc4_var label".split()
    
    # create an empty csv file
    file = open("/workspace/data/csv_files/data_3_sec.csv", "w", newline = "")
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    # labels name
    digits = "zero one two three four five six seven eight nine".split()
    
    # for each folder (zero => nine)...
    for nb in digits:

        # for each file...
        for filename in os.listdir(f"/workspace/data/audio_files/{nb}/"):
            
            # sound to process
            sound_name = f"/workspace/data/audio_files/{nb}/{filename}"
            
            # feature extraction
            y, sr = librosa.load(sound_name, mono = True, duration = 3)
            chroma_stft = librosa.feature.chroma_stft(y = y, sr = sr)
            rmse = librosa.feature.rms(y = y)
            spec_cent = librosa.feature.spectral_centroid(y = y, sr = sr)
            spec_bw = librosa.feature.spectral_bandwidth(y = y, sr = sr)
            rolloff = librosa.feature.spectral_rolloff(y = y, sr = sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y = y, sr = sr)
            
            # fill in the csv file
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {nb}'
            
            file = open('/workspace/data/csv_files/data_3_sec.csv', 'a', newline = '')

            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
            
            # transform csv file into dataframe
            df = pd.read_csv('/workspace/data/csv_files/data_3_sec.csv')
                
    return df


# main
if __name__ == '__main__':
    
    # status: python librairies importation 
    print("Import dependencies...\n")
    
    # status: start of data preprocessing 
    print("Dataframe is being created...\n")
    
    #create the dataframe with librosa parameters
    dataframe = createDataframe()
    
    # check 
    print(dataframe.head())
    
    # status: end of data preprocessing
    print("Data processing is finished!\n")
