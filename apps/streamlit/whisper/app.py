import numpy as np
import streamlit as st
import sys
import torch
import whisper
import librosa
import os
import io


@st.cache_resource
def load_model(model_id, model_path):
    # Check if two command-line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python app.py <whisper_model_id> <whisper_model_output_path>")
        print("Example: python app.py large-v3 /workspace/whisper-model/")
        sys.exit(1)

    # Check if the model path ends with '/'
    if not model_path.endswith('/'):
        model_path += '/'

    # Get complete model file path
    model_path = f'{model_path}{model_id}/{model_id}.pt'

    # Define available device (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model on available device
    model = whisper.load_model(model_path, device=device)

    # Display model's parameters in the app's logs
    print(
        f"Model will be run on {device}\n"
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )

    return model


def main():
    # Display Ttitle                
    st.title("Whisper - Speech to Text App")
                                            
    # Get args                              
    model_id = os.environ.get('MODEL_ID')   
    model_path = os.environ.get('MODEL_PATH')
                                             
    model = load_model(model_id, model_path) 
                                             
    # Upload audio file widget               
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
                                                                                                                                    
    transcript = {"text": "The audio file could not be transcribed :("}     
    options = dict(beam_size=5, best_of=5)                 
    transcribe_options = dict(task="transcribe", **options)          
                                                                     
    # Audio player                                                   
    if audio_file:                                           
        # Read file content                                
        audio_file = audio_file.read()
        st.audio(audio_file)          
                                      
        # Convert bytes to a file-like object using io.BytesIO
        audio_file = io.BytesIO(audio_file)                   
                                                              
        # Convert to numpy array                              
        audio_file, _ = librosa.load(audio_file)              
                                                
        # Transcribe audio on button click      
        if st.button("Transcribe"):             
            with st.spinner("Transcribing audio..."):
                transcript = model.transcribe(audio_file, **transcribe_options)
            st.write(transcript["text"])                                       


if __name__ == "__main__":                                                     
    main()
