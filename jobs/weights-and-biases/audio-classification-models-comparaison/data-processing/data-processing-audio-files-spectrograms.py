# import dependencies
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa
import numpy as np
from pathlib import Path
import librosa.display
import os
import splitfolders


# spectrograms generation function
def createSpectrograms(fold, labels):
    
    # folder where the spectrograms will be found
    spectrogram_path = Path('/workspace/data/spectrograms/')  
    
    # audio files location
    audio_path = Path('/workspace/data/audio_files/') 
    
    # display the folder (zero => nine) being processed
    print(f'Processing fold {fold}')
    
    # create each folder (zero => nine) for spectrogramms
    os.mkdir(spectrogram_path/fold)
    
    # spectrogram generation for each sound
    for audio_file in list(Path(audio_path/f'{fold}').glob('*.wav')):
        samples, sample_rate = librosa.load(audio_file)
        fig = plt.figure(figsize=[0.72,0.72])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        filename  = spectrogram_path/fold/Path(audio_file).name.replace('.wav','.png')
        S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
        plt.close('all')
        
    return S
        
    
# main
if __name__ == '__main__':
    
    # labels list
    classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    # call function for each folder (zero => nine)...
    for folder in classes:
        createSpectrograms(folder, classes)
    
    # split your data into a training and a validation set
    splitfolders.ratio("/workspace/data/spectrograms", output="/workspace/data/spectrograms_split", seed=1337, ratio=(.8, .2), group_prefix=None, move=False) # default values
