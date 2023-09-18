# import dependencies

# Audio Manipulation
import audioread
import librosa
from pydub import AudioSegment, silence
import youtube_dl
from youtube_dl import DownloadError

# Models
import torch
from transformers import pipeline, HubertForCTC, T5Tokenizer, T5ForConditionalGeneration, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
from pyannote.audio import Pipeline

# Others
from datetime import timedelta
import os
import pandas as pd
import pickle
import re
import streamlit as st
import time


def config():
    """
    App Configuration
    This functions sets the page title, its favicon, initialize some global variables (session_state values), displays
    a title, a smaller one, and apply CSS Code to the app.
    """
    # Set config
    st.set_page_config(page_title="Speech to Text", page_icon="📝")

    # Create a Data Directory
    # Will not be executed with AI Deploy because it is indicated in the DockerFile of the app

    if not os.path.exists("../data"):
        os.makedirs("../data")

    # Initialize session state variables
    if 'page_index' not in st.session_state:
        st.session_state['page_index'] = -1  # Handle which page should be displayed (token page, home page, results page, rename page)
        st.session_state['txt_transcript'] = ""  # Save the transcript as .txt so we can display it again on the results page
        st.session_state["process"] = []  # Save the results obtained so we can display them again on the results page
        st.session_state['srt_txt'] = ""  # Save the transcript in a subtitles case to display it on the results page
        st.session_state['srt_token'] = 0  # Is subtitles parameter enabled or not
        st.session_state['audio_file'] = None  # Save the audio file provided by the user so we can display it again on the results page
        st.session_state["start_time"] = 0  # Default audio player starting point (0s)
        st.session_state["summary"] = ""  # Save the summary of the transcript so we can display it on the results page
        st.session_state["number_of_speakers"] = 0  # Save the number of speakers detected in the conversation (diarization)
        st.session_state["chosen_mode"] = 0  # Save the mode chosen by the user (Diarization or not, timestamps or not)
        st.session_state["btn_token_list"] = []  # List of tokens that indicates what options are activated to adapt the display on results page
        st.session_state["my_HF_token"] = "ACCESS_TOKEN_GOES_HERE"  # User's Token that allows the use of the diarization model
        st.session_state["disable"] = True  # Default appearance of the button to change your token

    # Display Text and CSS
    st.title("Speech to Text App 📝")

    st.markdown("""
                    <style>
                    .block-container.css-12oz5g7.egzxvld2{
                        padding: 1%;}
                    # speech-to-text-app > div:nth-child(1) > span:nth-child(2){
                        text-align:center;}
                    .stRadio > label:nth-child(1){
                        font-weight: bold;
                        }
                    .stRadio > div{flex-direction:row;}
                    p, span{ 
                        text-align: justify;
                    }
                    span{ 
                        text-align: center;
                    }
                    """, unsafe_allow_html=True)

    st.subheader("You want to extract text from an audio/video? You are in the right place!")


def load_options(audio_length, dia_pipeline):
    """
    Display options so the user can customize the result (punctuate, summarize the transcript ? trim the audio? ...)
    User can choose his parameters thanks to sliders & checkboxes, both displayed in a st.form so the page doesn't
    reload when interacting with an element (frustrating if it does because user loses fluidity).
    :return: the chosen parameters
    """
    # Create a st.form()
    with st.form("form"):
        st.markdown("""<h6>
            You can transcript a specific part of your audio by setting start and end values below (in seconds). Then, 
            choose your parameters.</h6>""", unsafe_allow_html=True)

        # Possibility to trim / cut the audio on a specific part (=> transcribe less seconds will result in saving time)
        # To perform that, user selects his time intervals thanks to sliders, displayed in 2 different columns
        col1, col2 = st.columns(2)
        with col1:
            start = st.slider("Start value (s)", 0, audio_length, value=0)
        with col2:
            end = st.slider("End value (s)", 0, audio_length, value=audio_length)

        # Create 3 new columns to displayed other options
        col1, col2, col3 = st.columns(3)

        # User selects his preferences with checkboxes
        with col1:
            # Get an automatic punctuation
            punctuation_token = st.checkbox("Punctuate my final text", value=True)

            # Differentiate Speakers
            if dia_pipeline == None:
                st.write("Diarization model unvailable")
                diarization_token = False
            else:
                diarization_token = st.checkbox("Differentiate speakers")

        with col2:
            # Summarize the transcript
            summarize_token = st.checkbox("Generate a summary", value=False)

            # Generate a SRT file instead of a TXT file (shorter timestamps)
            srt_token = st.checkbox("Generate subtitles file", value=False)

        with col3:
            # Display the timestamp of each transcribed part
            timestamps_token = st.checkbox("Show timestamps", value=True)

            # Improve transcript with an other model (better transcript but longer to obtain)
            choose_better_model = st.checkbox("Change STT Model")

        # Srt option requires timestamps so it can matches text with time => Need to correct the following case
        if not timestamps_token and srt_token:
            timestamps_token = True
            st.warning("Srt option requires timestamps. We activated it for you.")

        # Validate choices with a button
        transcript_btn = st.form_submit_button("Transcribe audio!")

    return transcript_btn, start, end, diarization_token, punctuation_token, timestamps_token, srt_token, summarize_token, choose_better_model


@st.cache(allow_output_mutation=True)
def load_models():
    """
    Instead of systematically downloading each time the models we use (transcript model, summarizer, speaker differentiation, ...)
    thanks to transformers' pipeline, we first try to directly import them locally to save time when the app is launched.
    This function has a st.cache(), because as the models never change, we want the function to execute only one time
    (also to save time). Otherwise, it would run every time we transcribe a new audio file.
    :return: Loaded models
    """

    # Load facebook-hubert-large-ls960-ft model (English speech to text model)
    with st.spinner("Loading Speech to Text Model"):
        # If models are stored in a folder, we import them. Otherwise, we import the models with their respective library

        try:
            stt_tokenizer = pickle.load(open("models/STT_processor_hubert-large-ls960-ft.sav", 'rb'))
        except FileNotFoundError:
            stt_tokenizer = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

        try:
            stt_model = pickle.load(open("models/STT_model_hubert-large-ls960-ft.sav", 'rb'))
        except FileNotFoundError:
            stt_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

    # Load T5 model (Auto punctuation model)
    with st.spinner("Loading Punctuation Model"):
        try:
            t5_tokenizer = torch.load("models/T5_tokenizer.sav")
        except OSError:
            t5_tokenizer = T5Tokenizer.from_pretrained("flexudy/t5-small-wav2vec2-grammar-fixer")

        try:
            t5_model = torch.load("models/T5_model.sav")
        except FileNotFoundError:
            t5_model = T5ForConditionalGeneration.from_pretrained("flexudy/t5-small-wav2vec2-grammar-fixer")

    # Load summarizer model
    with st.spinner("Loading Summarization Model"):
        try:
            summarizer = pickle.load(open("models/summarizer.sav", 'rb'))
        except FileNotFoundError:
            summarizer = pipeline("summarization")

    # Load Diarization model (Differentiate speakers)
    with st.spinner("Loading Diarization Model"):
        try:
            dia_pipeline = pickle.load(open("models/dia_pipeline.sav", 'rb'))
        except FileNotFoundError:
            dia_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                                    use_auth_token=st.session_state["my_HF_token"])
            # If the token hasn't been modified, dia_pipeline will automatically be set to None. The functionality will then be disabled.

    return stt_tokenizer, stt_model, t5_tokenizer, t5_model, summarizer, dia_pipeline


def transcript_from_url(stt_tokenizer, stt_model, t5_tokenizer, t5_model, summarizer, dia_pipeline):
    """
    Display a text input area, where the user can enter a YouTube URL link. If the link seems correct, we try to
    extract the audio from the video, and then transcribe it.
    :param stt_tokenizer: Speech to text model's tokenizer
    :param stt_model: Speech to text model
    :param t5_tokenizer: Auto punctuation model's tokenizer
    :param t5_model: Auto punctuation model
    :param summarizer: Summarizer model
    :param dia_pipeline: Diarization Model (Differentiate speakers)
    """

    url = st.text_input("Enter the YouTube video URL then press Enter to confirm!")
    # If link seems correct, we try to transcribe
    if "youtu" in url:
        filename = extract_audio_from_yt_video(url)
        if filename is not None:
            transcription(stt_tokenizer, stt_model, t5_tokenizer, t5_model, summarizer, dia_pipeline, filename)
        else:
            st.error("We were unable to extract the audio. Please verify your link, retry or choose another video")


def transcript_from_file(stt_tokenizer, stt_model, t5_tokenizer, t5_model, summarizer, dia_pipeline):
    """
    Display a file uploader area, where the user can import his own file (mp3, mp4 or wav). If the file format seems
    correct, we transcribe the audio.
    :param stt_tokenizer: Speech to text model's tokenizer
    :param stt_model: Speech to text model
    :param t5_tokenizer: Auto punctuation model's tokenizer
    :param t5_model: Auto punctuation model
    :param summarizer: Summarizer model
    :param dia_pipeline: Diarization Model (Differentiate speakers)
    """

    # File uploader widget with a callback function, so the page reloads if the users uploads a new audio file
    uploaded_file = st.file_uploader("Upload your file! It can be a .mp3, .mp4 or .wav", type=["mp3", "mp4", "wav"],
                                     on_change=update_session_state, args=("page_index", 0,))

    if uploaded_file is not None:
        # get name and launch transcription function
        filename = uploaded_file.name
        transcription(stt_tokenizer, stt_model, t5_tokenizer, t5_model, summarizer, dia_pipeline, filename,
                      uploaded_file)


def transcription(stt_tokenizer, stt_model, t5_tokenizer, t5_model, summarizer, dia_pipeline, filename,
                  uploaded_file=None):
    """
    Mini-main function
    Display options, transcribe an audio file and save results.
    :param stt_tokenizer: Speech to text model's tokenizer
    :param stt_model: Speech to text model
    :param t5_tokenizer: Auto punctuation model's tokenizer
    :param t5_model: Auto punctuation model
    :param summarizer: Summarizer model
    :param dia_pipeline: Diarization Model (Differentiate speakers)
    :param filename: name of the audio file
    :param uploaded_file: file / name of the audio file which allows the code to reach the file
    """

    # If the audio comes from the Youtube extraction mode, the audio is downloaded so the uploaded_file is
    # the same as the filename. We need to change the uploaded_file which is currently set to None
    if uploaded_file is None:
        uploaded_file = filename

    # Get audio length of the file(s)
    myaudio = AudioSegment.from_file(uploaded_file)
    audio_length = myaudio.duration_seconds

    # Save Audio (so we can display it on another page ("DISPLAY RESULTS"), otherwise it is lost)
    update_session_state("audio_file", uploaded_file)

    # Display audio file
    st.audio(uploaded_file)

    # Is transcription possible
    if audio_length > 0:

        # We display options and user shares his wishes
        transcript_btn, start, end, diarization_token, punctuation_token, timestamps_token, srt_token, summarize_token, choose_better_model = load_options(
            int(audio_length), dia_pipeline)

        # If end value hasn't been changed, we fix it to the max value so we don't cut some ms of the audio because
        # end value is returned by a st.slider which return end value as a int (ex: return 12 sec instead of end=12.9s)
        if end == int(audio_length):
            end = audio_length

        # Switching model for the better one
        if choose_better_model:
            with st.spinner("We are loading the better model. Please wait..."):

                try:
                    stt_tokenizer = pickle.load(open("models/STT_tokenizer2_wav2vec2-large-960h-lv60-self.sav", 'rb'))
                except FileNotFoundError:
                    stt_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

                try:
                    stt_model = pickle.load(open("models/STT_model2_wav2vec2-large-960h-lv60-self.sav", 'rb'))
                except FileNotFoundError:
                    stt_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

        # Validate options and launch the transcription process thanks to the form's button
        if transcript_btn:

            # Check if start & end values are correct
            start, end = correct_values(start, end, audio_length)

            # If start a/o end value(s) has/have changed, we trim/cut the audio according to the new start/end values.
            if start != 0 or end != audio_length:
                myaudio = myaudio[start * 1000:end * 1000]  # Works in milliseconds (*1000)

            # Transcribe process is running
            with st.spinner("We are transcribing your audio. Please wait"):

                # Initialize variables
                txt_text, srt_text, save_result = init_transcription(start, int(end))
                min_space, max_space = silence_mode_init(srt_token)

                # Differentiate speakers mode
                if diarization_token:

                    # Save mode chosen by user, to display expected results
                    if not timestamps_token:
                        update_session_state("chosen_mode", "DIA")
                    elif timestamps_token:
                        update_session_state("chosen_mode", "DIA_TS")

                    # Convert mp3/mp4 to wav (Differentiate speakers mode only accepts wav files)
                    if filename.endswith((".mp3", ".mp4")):
                        myaudio, filename = convert_file_to_wav(myaudio, filename)
                    else:
                        filename = "../data/" + filename
                        myaudio.export(filename, format="wav")

                    # Differentiate speakers process
                    diarization_timestamps, number_of_speakers = diarization_treatment(filename, dia_pipeline,
                                                                                       max_space, srt_token)
                    # Saving the number of detected speakers
                    update_session_state("number_of_speakers", number_of_speakers)

                    # Transcribe process with Diarization Mode
                    save_result, txt_text, srt_text = transcription_diarization(filename, diarization_timestamps,
                                                                                stt_model,
                                                                                stt_tokenizer,
                                                                                diarization_token,
                                                                                srt_token, summarize_token,
                                                                                timestamps_token, myaudio, start,
                                                                                save_result,
                                                                                txt_text, srt_text)

                # Non Diarization Mode
                else:
                    # Save mode chosen by user, to display expected results
                    if not timestamps_token:
                        update_session_state("chosen_mode", "NODIA")
                    if timestamps_token:
                        update_session_state("chosen_mode", "NODIA_TS")

                    filename = "../data/" + filename
                    # Transcribe process with non Diarization Mode
                    save_result, txt_text, srt_text = transcription_non_diarization(filename, myaudio, start, end,
                                                                                    diarization_token, timestamps_token,
                                                                                    srt_token, summarize_token,
                                                                                    stt_model, stt_tokenizer,
                                                                                    min_space, max_space,
                                                                                    save_result, txt_text, srt_text)

                # Save results so it is not lost when we interact with a button
                update_session_state("process", save_result)
                update_session_state("srt_txt", srt_text)

                # Get final text (with or without punctuation token)
                # Diariation Mode
                if diarization_token:
                    # Create txt text from the process
                    txt_text = create_txt_text_from_process(punctuation_token, t5_model, t5_tokenizer)

                # Non diarization Mode
                else:

                    if punctuation_token:
                        # Need to split the text by 512 text blocks size since the model has a limited input
                        with st.spinner("Transcription is finished! Let us punctuate your audio"):
                            my_split_text_list = split_text(txt_text, 512)
                            txt_text = ""
                            # punctuate each text block
                            for my_split_text in my_split_text_list:
                                txt_text += add_punctuation(t5_model, t5_tokenizer, my_split_text)

                # Clean folder's files
                clean_directory("../data")

                # Display the final transcript
                if txt_text != "":
                    st.subheader("Final text is")

                    # Save txt_text and display it
                    update_session_state("txt_transcript", txt_text)
                    st.markdown(txt_text, unsafe_allow_html=True)

                    # Summarize the transcript
                    if summarize_token:
                        with st.spinner("We are summarizing your audio"):
                            # Display summary in a st.expander widget to don't write too much text on the page
                            with st.expander("Summary"):
                                # Need to split the text by 1024 text blocks size since the model has a limited input
                                if diarization_token:
                                    # in diarization mode, the text to summarize is contained in the "summary" the session state variable
                                    my_split_text_list = split_text(st.session_state["summary"], 1024)
                                else:
                                    # in non-diarization mode, it is contained in the txt_text variable
                                    my_split_text_list = split_text(txt_text, 1024)

                                summary = ""
                                # Summarize each text block
                                for my_split_text in my_split_text_list:
                                    summary += summarizer(my_split_text)[0]['summary_text']

                                # Removing multiple spaces and double spaces around punctuation mark " . "
                                summary = re.sub(' +', ' ', summary)
                                summary = re.sub(r'\s+([?.!"])', r'\1', summary)

                                # Display summary and save it
                                st.write(summary)
                                update_session_state("summary", summary)

                    # Display buttons to interact with results

                    # We have 4 possible buttons depending on the user's choices. But we can't set 4 columns for 4
                    # buttons. Indeed, if the user displays only 3 buttons, it is possible that one of the column
                    # 1, 2 or 3 is empty which would be ugly. We want the activated options to be in the first columns
                    # so that the empty columns are not noticed. To do that, let's create a btn_token_list

                    btn_token_list = [[diarization_token, "dia_token"], [True, "useless_txt_token"],
                                      [srt_token, "srt_token"], [summarize_token, "summarize_token"]]

                    # Save this list to be able to reach it on the other pages of the app
                    update_session_state("btn_token_list", btn_token_list)

                    # Create 4 columns
                    col1, col2, col3, col4 = st.columns(4)

                    # Create a column list
                    col_list = [col1, col2, col3, col4]

                    # Check value of each token, if True, we put the respective button of the token in a column
                    col_index = 0
                    for elt in btn_token_list:
                        if elt[0]:
                            mycol = col_list[col_index]
                            if elt[1] == "useless_txt_token":
                                # Download your transcript.txt
                                with mycol:
                                    st.download_button("Download as TXT", txt_text, file_name="my_transcription.txt",
                                                       on_click=update_session_state, args=("page_index", 1,))
                            elif elt[1] == "srt_token":
                                # Download your transcript.srt
                                with mycol:
                                    update_session_state("srt_token", srt_token)
                                    st.download_button("Download as SRT", srt_text, file_name="my_transcription.srt",
                                                       on_click=update_session_state, args=("page_index", 1,))
                            elif elt[1] == "dia_token":
                                with mycol:
                                    # Rename the speakers detected in your audio
                                    st.button("Rename Speakers", on_click=update_session_state, args=("page_index", 2,))

                            elif elt[1] == "summarize_token":
                                with mycol:
                                    # Download the summary of your transcript.txt
                                    st.download_button("Download Summary", st.session_state["summary"],
                                                       file_name="my_summary.txt",
                                                       on_click=update_session_state, args=("page_index", 1,))
                            col_index += 1

                else:
                    st.write("Transcription impossible, a problem occurred with your audio or your parameters, "
                             "we apologize :(")

    else:
        st.error("Seems your audio is 0 s long, please change your file")
        time.sleep(3)
        st.stop()


def create_txt_text_from_process(punctuation_token=False, t5_model=None, t5_tokenizer=None):
    """
    If we are in a diarization case (differentiate speakers), we create txt_text from st.session.state['process']
    There is a lot of information in the process variable, but we only extract the identity of the speaker and
    the sentence spoken, as in a non-diarization case.
    :param punctuation_token: Punctuate or not the transcript (choice fixed by user)
    :param t5_model: T5 Model (Auto punctuation model)
    :param t5_tokenizer: T5’s Tokenizer (Auto punctuation model's tokenizer)
    :return: Final transcript (without timestamps)
    """
    txt_text = ""
    # The information to be extracted is different according to the chosen mode
    if punctuation_token:
        with st.spinner("Transcription is finished! Let us punctuate your audio"):
            if st.session_state["chosen_mode"] == "DIA":
                for elt in st.session_state["process"]:
                    # [2:] don't want ": text" but only the "text"
                    text_to_punctuate = elt[2][2:]
                    if len(text_to_punctuate) >= 512:
                        text_to_punctutate_list = split_text(text_to_punctuate, 512)
                        punctuated_text = ""
                        for split_text_to_punctuate in text_to_punctutate_list:
                            punctuated_text += add_punctuation(t5_model, t5_tokenizer, split_text_to_punctuate)
                    else:
                        punctuated_text = add_punctuation(t5_model, t5_tokenizer, text_to_punctuate)

                    txt_text += elt[1] + " : " + punctuated_text + '\n\n'

            elif st.session_state["chosen_mode"] == "DIA_TS":
                for elt in st.session_state["process"]:
                    text_to_punctuate = elt[3][2:]
                    if len(text_to_punctuate) >= 512:
                        text_to_punctutate_list = split_text(text_to_punctuate, 512)
                        punctuated_text = ""
                        for split_text_to_punctuate in text_to_punctutate_list:
                            punctuated_text += add_punctuation(t5_model, t5_tokenizer, split_text_to_punctuate)
                    else:
                        punctuated_text = add_punctuation(t5_model, t5_tokenizer, text_to_punctuate)

                    txt_text += elt[2] + " : " + punctuated_text + '\n\n'
    else:
        if st.session_state["chosen_mode"] == "DIA":
            for elt in st.session_state["process"]:
                txt_text += elt[1] + elt[2] + '\n\n'

        elif st.session_state["chosen_mode"] == "DIA_TS":
            for elt in st.session_state["process"]:
                txt_text += elt[2] + elt[3] + '\n\n'

    return txt_text


def rename_speakers_window():
    """
    Load a new page which allows the user to rename the different speakers from the diarization process
    For example he can switch from "Speaker1 : "I wouldn't say that"" to "Mat : "I wouldn't say that""
    """

    st.subheader("Here you can rename the speakers as you want")
    number_of_speakers = st.session_state["number_of_speakers"]

    if number_of_speakers > 0:
        # Handle displayed text according to the number_of_speakers
        if number_of_speakers == 1:
            st.write(str(number_of_speakers) + " speaker has been detected in your audio")
        else:
            st.write(str(number_of_speakers) + " speakers have been detected in your audio")

        # Saving the Speaker Name and its ID in a list, example : [1, 'Speaker1']
        list_of_speakers = []
        for elt in st.session_state["process"]:
            if st.session_state["chosen_mode"] == "DIA_TS":
                if [elt[1], elt[2]] not in list_of_speakers:
                    list_of_speakers.append([elt[1], elt[2]])
            elif st.session_state["chosen_mode"] == "DIA":
                if [elt[0], elt[1]] not in list_of_speakers:
                    list_of_speakers.append([elt[0], elt[1]])

        # Sorting (by ID)
        list_of_speakers.sort()  # [[1, 'Speaker1'], [0, 'Speaker0']] => [[0, 'Speaker0'], [1, 'Speaker1']]

        # Display saved names so the user can modify them
        initial_names = ""
        for elt in list_of_speakers:
            initial_names += elt[1] + "\n"

        names_input = st.text_area("Just replace the names without changing the format (one per line)",
                                   value=initial_names)

        # Display Options (Cancel / Save)
        col1, col2 = st.columns(2)
        with col1:
            # Cancel changes by clicking a button - callback function to return to the results page
            st.button("Cancel", on_click=update_session_state, args=("page_index", 1,))
        with col2:
            # Confirm changes by clicking a button - callback function to apply changes and return to the results page
            st.button("Save changes", on_click=click_confirm_rename_btn, args=(names_input, number_of_speakers,))

    # Don't have anyone to rename
    else:
        st.error("0 speakers have been detected. Seem there is an issue with diarization")
        with st.spinner("Redirecting to transcription page"):
            time.sleep(4)
            # return to the results page
            update_session_state("page_index", 1)


def click_confirm_rename_btn(names_input, number_of_speakers):
    """
    If the users decides to rename speakers and confirms his choices, we apply the modifications to our transcript
    Then we return to the results page of the app
    :param names_input: string
    :param number_of_speakers: Number of detected speakers in the audio file
    """

    try:
        names_input = names_input.split("\n")[:number_of_speakers]

        for elt in st.session_state["process"]:
            elt[2] = names_input[elt[1]]

        txt_text = create_txt_text_from_process()
        update_session_state("txt_transcript", txt_text)
        update_session_state("page_index", 1)

    except TypeError:  # list indices must be integers or slices, not str (happened to me one time when writing non sense names)
        st.error("Please respect the 1 name per line format")
        with st.spinner("We are relaunching the page"):
            time.sleep(3)
            update_session_state("page_index", 1)


def transcription_diarization(filename, diarization_timestamps, stt_model, stt_tokenizer, diarization_token, srt_token,
                              summarize_token, timestamps_token, myaudio, start, save_result, txt_text, srt_text):
    """
    Performs transcription with the diarization mode
    :param filename: name of the audio file
    :param diarization_timestamps: timestamps of each audio part (ex 10 to 50 secs)
    :param stt_model: Speech to text model
    :param stt_tokenizer: Speech to text model's tokenizer
    :param diarization_token: Differentiate or not the speakers (choice fixed by user)
    :param srt_token: Enable/Disable generate srt file (choice fixed by user)
    :param summarize_token: Summarize or not the transcript (choice fixed by user)
    :param timestamps_token: Display and save or not the timestamps (choice fixed by user)
    :param myaudio: AudioSegment file
    :param start: int value (s) given by st.slider() (fixed by user)
    :param save_result: whole process
    :param txt_text: generated .txt transcript
    :param srt_text: generated .srt transcript
    :return: results of transcribing action
    """
    # Numeric counter that identifies each sequential subtitle
    srt_index = 1

    # Handle a rare case : Only the case if only one "list" in the list (it makes a classic list) not a list of list
    if not isinstance(diarization_timestamps[0], list):
        diarization_timestamps = [diarization_timestamps]

    # Transcribe each audio chunk (from timestamp to timestamp) and display transcript
    for index, elt in enumerate(diarization_timestamps):
        sub_start = elt[0]
        sub_end = elt[1]

        transcription = transcribe_audio_part(filename, stt_model, stt_tokenizer, myaudio, sub_start, sub_end,
                                              index)

        # Initial audio has been split with start & end values
        # It begins to 0s, but the timestamps need to be adjust with +start*1000 values to adapt the gap
        if transcription != "":
            save_result, txt_text, srt_text, srt_index = display_transcription(diarization_token, summarize_token,
                                                                    srt_token, timestamps_token,
                                                                    transcription, save_result, txt_text,
                                                                    srt_text,
                                                                    srt_index, sub_start + start * 1000,
                                                                    sub_end + start * 1000, elt)
    return save_result, txt_text, srt_text


def transcription_non_diarization(filename, myaudio, start, end, diarization_token, timestamps_token, srt_token,
                                  summarize_token, stt_model, stt_tokenizer, min_space, max_space, save_result,
                                  txt_text, srt_text):
    """
    Performs transcribing action with the non-diarization mode
    :param filename: name of the audio file
    :param myaudio: AudioSegment file
    :param start: int value (s) given by st.slider() (fixed by user)
    :param end: int value (s) given by st.slider() (fixed by user)
    :param diarization_token: Differentiate or not the speakers (choice fixed by user)
    :param timestamps_token: Display and save or not the timestamps (choice fixed by user)
    :param srt_token: Enable/Disable generate srt file (choice fixed by user)
    :param summarize_token: Summarize or not the transcript (choice fixed by user)
    :param stt_model: Speech to text model
    :param stt_tokenizer: Speech to text model's tokenizer
    :param min_space: Minimum temporal distance between two silences
    :param max_space: Maximum temporal distance between two silences
    :param save_result: whole process
    :param txt_text: generated .txt transcript
    :param srt_text: generated .srt transcript
    :return: results of transcribing action
    """

    # Numeric counter identifying each sequential subtitle
    srt_index = 1

    # get silences
    silence_list = detect_silences(myaudio)
    if silence_list != []:
        silence_list = get_middle_silence_time(silence_list)
        silence_list = silences_distribution(silence_list, min_space, max_space, start, end, srt_token)
    else:
        silence_list = generate_regular_split_till_end(silence_list, int(end), min_space, max_space)

    # Transcribe each audio chunk (from timestamp to timestamp) and display transcript
    for i in range(0, len(silence_list) - 1):
        sub_start = silence_list[i]
        sub_end = silence_list[i + 1]

        transcription = transcribe_audio_part(filename, stt_model, stt_tokenizer, myaudio, sub_start, sub_end, i)

        # Initial audio has been split with start & end values
        # It begins to 0s, but the timestamps need to be adjust with +start*1000 values to adapt the gap
        if transcription != "":
            save_result, txt_text, srt_text, srt_index = display_transcription(diarization_token, summarize_token,
                                                                    srt_token, timestamps_token,
                                                                    transcription, save_result,
                                                                    txt_text,
                                                                    srt_text,
                                                                    srt_index, sub_start + start * 1000,
                                                                    sub_end + start * 1000)

    return save_result, txt_text, srt_text


def silence_mode_init(srt_token):
    """
    Fix min_space and max_space values
    If the user wants a srt file, we need to have tiny timestamps
    :param srt_token: Enable/Disable generate srt file option (choice fixed by user)
    :return: min_space and max_space values
    """
    if srt_token:
        # We need short intervals if we want a short text
        min_space = 1000  # 1 sec
        max_space = 8000  # 8 secs

    else:
        min_space = 25000  # 25 secs
        max_space = 45000  # 45secs

    return min_space, max_space


def detect_silences(audio):
    """
    Silence moments detection in an audio file
    :param audio: pydub.AudioSegment file
    :return: list with silences time intervals
    """
    # Get Decibels (dB) so silences detection depends on the audio instead of a fixed value
    dbfs = audio.dBFS

    # Get silences timestamps > 750ms
    silence_list = silence.detect_silence(audio, min_silence_len=750, silence_thresh=dbfs - 14)

    return silence_list


def generate_regular_split_till_end(time_list, end, min_space, max_space):
    """
    Add automatic "time cuts" to time_list till end value depending on min_space and max_space values
    :param time_list: silence time list
    :param end: int value (s)
    :param min_space: Minimum temporal distance between two silences
    :param max_space: Maximum temporal distance between two silences
    :return: list with automatic time cuts
    """
    # In range loop can't handle float values so we convert to int
    int_last_value = int(time_list[-1])
    int_end = int(end)

    # Add maxspace to the last list value and add this value to the list
    for i in range(int_last_value, int_end, max_space):
        value = i + max_space
        if value < end:
            time_list.append(value)

    # Fix last automatic cut
    # If small gap (ex: 395 000, with end = 400 000)
    if end - time_list[-1] < min_space:
        time_list[-1] = end
    else:
        # If important gap (ex: 311 000 then 356 000, with end = 400 000, can't replace and then have 311k to 400k)
        time_list.append(end)
    return time_list


def get_middle_silence_time(silence_list):
    """
    Replace in a list each timestamp by a unique value, which is approximately the middle of each silence timestamp, to
    avoid word cutting
    :param silence_list: List of lists where each element has a start and end value which describes a silence timestamp
    :return: Simple float list
    """
    length = len(silence_list)
    index = 0
    while index < length:
        diff = (silence_list[index][1] - silence_list[index][0])
        if diff < 3500:
            silence_list[index] = silence_list[index][0] + diff / 2
            index += 1
        else:
            adapted_diff = 1500
            silence_list.insert(index + 1, silence_list[index][1] - adapted_diff)
            silence_list[index] = silence_list[index][0] + adapted_diff
            length += 1
            index += 2

    return silence_list


def silences_distribution(silence_list, min_space, max_space, start, end, srt_token=False):
    """
    We keep each silence value if it is sufficiently distant from its neighboring values, without being too much
    :param silence_list: List with silences intervals
    :param min_space: Minimum temporal distance between two silences
    :param max_space: Maximum temporal distance between two silences
    :param start: int value (seconds)
    :param end: int value (seconds)
    :param srt_token: Enable/Disable generate srt file (choice fixed by user)
    :return: list with equally distributed silences
    """
    # If starts != 0, we need to adjust end value since silences detection is performed on the trimmed/cut audio
    # (and not on the original audio) (ex: trim audio from 20s to 2m will be 0s to 1m40 = 2m-20s)

    # Shift the end according to the start value
    end -= start
    start = 0
    end *= 1000

    # Step 1 - Add start value
    newsilence = [start]

    # Step 2 - Create a regular distribution between start and the first element of silence_list to don't have a gap > max_space and run out of memory
    # example newsilence = [0] and silence_list starts with 100000 => It will create a massive gap [0, 100000]

    if silence_list[0] - max_space > newsilence[0]:
        for i in range(int(newsilence[0]), int(silence_list[0]), max_space):  # int bc float can't be in a range loop
            value = i + max_space
            if value < silence_list[0]:
                newsilence.append(value)

    # Step 3 - Create a regular distribution until the last value of the silence_list
    min_desired_value = newsilence[-1]
    max_desired_value = newsilence[-1]
    nb_values = len(silence_list)

    while nb_values != 0:
        max_desired_value += max_space

        # Get a window of the values greater than min_desired_value and lower than max_desired_value
        silence_window = list(filter(lambda x: min_desired_value < x <= max_desired_value, silence_list))

        if silence_window != []:
            # Get the nearest value we can to min_desired_value or max_desired_value depending on srt_token
            if srt_token:
                nearest_value = min(silence_window, key=lambda x: abs(x - min_desired_value))
                nb_values -= silence_window.index(nearest_value) + 1  # (index begins at 0, so we add 1)
            else:
                nearest_value = min(silence_window, key=lambda x: abs(x - max_desired_value))
                # Max value index = len of the list
                nb_values -= len(silence_window)

            # Append the nearest value to our list
            newsilence.append(nearest_value)

        # If silence_window is empty we add the max_space value to the last one to create an automatic cut and avoid multiple audio cutting
        else:
            newsilence.append(newsilence[-1] + max_space)

        min_desired_value = newsilence[-1]
        max_desired_value = newsilence[-1]

    # Step 4 - Add the final value (end)

    if end - newsilence[-1] > min_space:
        # Gap > Min Space
        if end - newsilence[-1] < max_space:
            newsilence.append(end)
        else:
            # Gap too important between the last list value and the end value
            # We need to create automatic max_space cut till the end
            newsilence = generate_regular_split_till_end(newsilence, end, min_space, max_space)
    else:
        # Gap < Min Space <=> Final value and last value of new silence are too close, need to merge
        if len(newsilence) >= 2:
            if end - newsilence[-2] <= max_space:
                # Replace if gap is not too important
                newsilence[-1] = end
            else:
                newsilence.append(end)

        else:
            if end - newsilence[-1] <= max_space:
                # Replace if gap is not too important
                newsilence[-1] = end
            else:
                newsilence.append(end)

    return newsilence


def init_transcription(start, end):
    """
    Initialize values and inform user that transcription is in progress
    :param start: int value (s) given by st.slider() (fixed by user)
    :param end: int value (s) given by st.slider() (fixed by user)
    :return: final_transcription, final_srt_text, and the process
    """
    update_session_state("summary", "")
    st.write("Transcription between", start, "and", end, "seconds in process.\n\n")
    txt_text = ""
    srt_text = ""
    save_result = []
    return txt_text, srt_text, save_result


def transcribe_audio_part(filename, stt_model, stt_tokenizer, myaudio, sub_start, sub_end, index):
    """
    Transcribe an audio between a sub_start and a sub_end value (s)
    :param filename: name of the audio file
    :param stt_model: Speech to text model
    :param stt_tokenizer: Speech to text model's tokenizer
    :param myaudio: AudioSegment file
    :param sub_start: start value (s) of the considered audio part to transcribe
    :param sub_end: end value (s) of the considered audio part to transcribe
    :param index: audio file counter
    :return: transcription of the considered audio (only in uppercase, so we add lower() to make the reading easier)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        with torch.no_grad():
            new_audio = myaudio[sub_start:sub_end]  # Works in milliseconds
            path = filename[:-3] + "audio_" + str(index) + ".wav"
            new_audio.export(path)  # Exports to a wav file in the current path

            # Load audio file with librosa, set sound rate to 16000 Hz because the model we use was trained on 16000 Hz data
            input_audio, _ = librosa.load(path, sr=16000)

            # return PyTorch torch.Tensor instead of a list of python integers thanks to return_tensors = ‘pt’
            input_values = stt_tokenizer(input_audio, sampling_rate=16000, return_tensors="pt").to(device).input_values

            # Get logits from the data structure containing all the information returned by the model and get our prediction
            logits = stt_model.to(device)(input_values).logits
            prediction = torch.argmax(logits, dim=-1)

            # Decode & lower our string (model's output is only uppercase)
            if isinstance(stt_tokenizer, Wav2Vec2Tokenizer):
                transcription = stt_tokenizer.batch_decode(prediction)[0]
            elif isinstance(stt_tokenizer, Wav2Vec2Processor):
                transcription = stt_tokenizer.decode(prediction[0])

            # return transcription
            return transcription.lower()

    except audioread.NoBackendError:
        # Means we have a chunk with a [value1 : value2] case with value1>value2
        st.error("Sorry, seems we have a problem on our side. Please change start & end values.")
        time.sleep(3)
        st.stop()


def optimize_subtitles(transcription, srt_index, sub_start, sub_end, srt_text):
    """
    Create & Optimize the subtitles (avoid a too long reading when many words are said in a short time)
    The optimization (if statement) can sometimes create a gap between the subtitles and the video, if there is music
    for example. In this case, it may be wise to disable the optimization, never going through the if statement.
    :param transcription: transcript generated for an audio chunk
    :param srt_index: Numeric counter that identifies each sequential subtitle
    :param sub_start: beginning of the transcript
    :param sub_end: end of the transcript
    :param srt_text: generated .srt transcript
    """

    transcription_length = len(transcription)

    # Length of the transcript should be limited to about 42 characters per line to avoid this problem
    if transcription_length > 42:
        # Split the timestamp and its transcript in two parts
        # Get the middle timestamp
        diff = (timedelta(milliseconds=sub_end) - timedelta(milliseconds=sub_start)) / 2
        middle_timestamp = str(timedelta(milliseconds=sub_start) + diff).split(".")[0]

        # Get the closest middle index to a space (we don't divide transcription_length/2 to avoid cutting a word)
        space_indexes = [pos for pos, char in enumerate(transcription) if char == " "]
        nearest_index = min(space_indexes, key=lambda x: abs(x - transcription_length / 2))

        # First transcript part
        first_transcript = transcription[:nearest_index]

        # Second transcript part
        second_transcript = transcription[nearest_index + 1:]

        # Add both transcript parts to the srt_text
        srt_text += str(srt_index) + "\n" + str(timedelta(milliseconds=sub_start)).split(".")[0] + " --> " + middle_timestamp + "\n" + first_transcript + "\n\n"
        srt_index += 1
        srt_text += str(srt_index) + "\n" + middle_timestamp + " --> " + str(timedelta(milliseconds=sub_end)).split(".")[0] + "\n" + second_transcript + "\n\n"
        srt_index += 1
    else:
        # Add transcript without operations
        srt_text += str(srt_index) + "\n" + str(timedelta(milliseconds=sub_start)).split(".")[0] + " --> " + str(timedelta(milliseconds=sub_end)).split(".")[0] + "\n" + transcription + "\n\n"

    return srt_text, srt_index


def display_transcription(diarization_token, summarize_token, srt_token, timestamps_token, transcription, save_result,
                          txt_text, srt_text, srt_index, sub_start, sub_end, elt=None):
    """
    Display results
    :param diarization_token: Differentiate or not the speakers (choice fixed by user)
    :param summarize_token: Summarize or not the transcript (choice fixed by user)
    :param srt_token: Enable/Disable generate srt file (choice fixed by user)
    :param timestamps_token: Display and save or not the timestamps (choice fixed by user)
    :param transcription: transcript of the considered audio
    :param save_result: whole process
    :param txt_text: generated .txt transcript
    :param srt_text: generated .srt transcript
    :param srt_index : numeric counter that identifies each sequential subtitle
    :param sub_start: start value (s) of the considered audio part to transcribe
    :param sub_end: end value (s) of the considered audio part to transcribe
    :param elt: timestamp (diarization case only, otherwise elt = None)
    """
    # Display will be different depending on the mode (dia, no dia, dia_ts, nodia_ts)
    # diarization mode
    if diarization_token:

        if summarize_token:
            update_session_state("summary", transcription + " ", concatenate_token=True)

        if not timestamps_token:
            temp_transcription = elt[2] + " : " + transcription
            st.write(temp_transcription + "\n\n")

            save_result.append([int(elt[2][-1]), elt[2], " : " + transcription])

        elif timestamps_token:
            temp_timestamps = str(timedelta(milliseconds=sub_start)).split(".")[0] + " --> " + \
                              str(timedelta(milliseconds=sub_end)).split(".")[0] + "\n"
            temp_transcription = elt[2] + " : " + transcription
            temp_list = [temp_timestamps, int(elt[2][-1]), elt[2], " : " + transcription, int(sub_start / 1000)]
            save_result.append(temp_list)
            st.button(temp_timestamps, on_click=click_timestamp_btn, args=(sub_start,))
            st.write(temp_transcription + "\n\n")

            if srt_token:
                srt_text, srt_index = optimize_subtitles(transcription, srt_index, sub_start, sub_end, srt_text)

    # Non diarization case
    else:
        if not timestamps_token:
            save_result.append([transcription])
            st.write(transcription + "\n\n")

        else:
            temp_timestamps = str(timedelta(milliseconds=sub_start)).split(".")[0] + " --> " + \
                              str(timedelta(milliseconds=sub_end)).split(".")[0] + "\n"
            temp_list = [temp_timestamps, transcription, int(sub_start / 1000)]
            save_result.append(temp_list)
            st.button(temp_timestamps, on_click=click_timestamp_btn, args=(sub_start,))
            st.write(transcription + "\n\n")

            if srt_token:
                srt_text, srt_index = optimize_subtitles(transcription, srt_index, sub_start, sub_end, srt_text)

        txt_text += transcription + " "  # So x seconds sentences are separated

    return save_result, txt_text, srt_text, srt_index


def add_punctuation(t5_model, t5_tokenizer, transcript):
    """
    Punctuate a transcript
    :return: Punctuated and improved (corrected) transcript
    """
    input_text = "fix: { " + transcript + " } </s>"

    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=10000, truncation=True,
                                    add_special_tokens=True)

    outputs = t5_model.generate(
        input_ids=input_ids,
        max_length=256,
        num_beams=4,
        repetition_penalty=1.0,
        length_penalty=1.0,
        early_stopping=True
    )

    transcript = t5_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return transcript


def convert_file_to_wav(aud_seg, filename):
    """
    Convert a mp3/mp4 in a wav format
    Needs to be modified if you want to convert a format which contains less or more than 3 letters
    :param aud_seg: pydub.AudioSegment
    :param filename: name of the file
    :return: name of the converted file
    """
    filename = "../data/my_wav_file_" + filename[:-3] + "wav"
    aud_seg.export(filename, format="wav")

    newaudio = AudioSegment.from_file(filename)

    return newaudio, filename


def get_diarization(dia_pipeline, filename):
    """
    Diarize an audio (find numbers of speakers, when they speak, ...)
    :param dia_pipeline: Pyannote's library (diarization pipeline)
    :param filename: name of a wav audio file
    :return: str list containing audio's diarization time intervals
    """
    # Get diarization of the audio
    diarization = dia_pipeline({'audio': filename})
    listmapping = diarization.labels()
    listnewmapping = []

    # Rename default speakers' names (Default is A, B, ...), we want Speaker0, Speaker1, ...
    number_of_speakers = len(listmapping)
    for i in range(number_of_speakers):
        listnewmapping.append("Speaker" + str(i))

    mapping_dict = dict(zip(listmapping, listnewmapping))

    diarization.rename_labels(mapping_dict,
                              copy=False)  # copy set to False so we don't create a new annotation, we replace the actual on

    return diarization, number_of_speakers


def confirm_token_change(hf_token, page_index):
    """
    A function that saves the hugging face token entered by the user.
    It also updates the page index variable so we can indicate we now want to display the home page instead of the token page
    :param hf_token: user's token
    :param page_index: number that represents the home page index (mentioned in the main.py file)
    """
    update_session_state("my_HF_token", hf_token)
    update_session_state("page_index", page_index)


def convert_str_diarlist_to_timedelta(diarization_result):
    """
    Extract from Diarization result the given speakers with their respective speaking times and transform them in pandas timedelta objects
    :param diarization_result: result of diarization
    :return: list with timedelta intervals and their respective speaker
    """

    # get speaking intervals from diarization
    segments = diarization_result.for_json()["content"]
    diarization_timestamps = []
    for sample in segments:
        # Convert segment in a pd.Timedelta object
        new_seg = [pd.Timedelta(seconds=round(sample["segment"]["start"], 2)),
                   pd.Timedelta(seconds=round(sample["segment"]["end"], 2)), sample["label"]]
        # Start and end = speaking duration
        # label = who is speaking
        diarization_timestamps.append(new_seg)

    return diarization_timestamps


def merge_speaker_times(diarization_timestamps, max_space, srt_token):
    """
    Merge near times for each detected speaker (Same speaker during 1-2s and 3-4s -> Same speaker during 1-4s)
    :param diarization_timestamps: diarization list
    :param max_space: Maximum temporal distance between two silences
    :param srt_token: Enable/Disable generate srt file (choice fixed by user)
    :return: list with timedelta intervals and their respective speaker
    """
    if not srt_token:
        threshold = pd.Timedelta(seconds=max_space / 1000)

        index = 0
        length = len(diarization_timestamps) - 1

        while index < length:
            if diarization_timestamps[index + 1][2] == diarization_timestamps[index][2] and \
                    diarization_timestamps[index + 1][1] - threshold <= diarization_timestamps[index][0]:
                diarization_timestamps[index][1] = diarization_timestamps[index + 1][1]
                del diarization_timestamps[index + 1]
                length -= 1
            else:
                index += 1
    return diarization_timestamps


def extending_timestamps(new_diarization_timestamps):
    """
    Extend timestamps between each diarization timestamp if possible, so we avoid word cutting
    :param new_diarization_timestamps: list
    :return: list with merged times
    """
    for i in range(1, len(new_diarization_timestamps)):
        if new_diarization_timestamps[i][0] - new_diarization_timestamps[i - 1][1] <= timedelta(milliseconds=3000) and \
                new_diarization_timestamps[i][0] - new_diarization_timestamps[i - 1][1] >= timedelta(milliseconds=100):
            middle = (new_diarization_timestamps[i][0] - new_diarization_timestamps[i - 1][1]) / 2
            new_diarization_timestamps[i][0] -= middle
            new_diarization_timestamps[i - 1][1] += middle

    # Converting list so we have a milliseconds format
    for elt in new_diarization_timestamps:
        elt[0] = elt[0].total_seconds() * 1000
        elt[1] = elt[1].total_seconds() * 1000

    return new_diarization_timestamps


def clean_directory(path):
    """
    Clean files of directory
    :param path: directory's path
    """
    for file in os.listdir(path):
        os.remove(os.path.join(path, file))


def correct_values(start, end, audio_length):
    """
    Start or/and end value(s) can be in conflict, so we check these values
    :param start: int value (s) given by st.slider() (fixed by user)
    :param end: int value (s) given by st.slider() (fixed by user)
    :param audio_length: audio duration (s)
    :return: approved values
    """
    # Start & end Values need to be checked

    if start >= audio_length or start >= end:
        start = 0
        st.write("Start value has been set to 0s because of conflicts with other values")

    if end > audio_length or end == 0:
        end = audio_length
        st.write("End value has been set to maximum value because of conflicts with other values")

    return start, end


def split_text(my_text, max_size):
    """
    Split a text
    Maximum sequence length for this model is max_size.
    If the transcript is longer, it needs to be split by the nearest possible value to max_size.
    To avoid cutting words, we will cut on "." characters, and " " if there is not "."
    :return: split text
    """

    cut2 = max_size

    # First, we get indexes of "."
    my_split_text_list = []
    nearest_index = 0
    length = len(my_text)
    # We split the transcript in text blocks of size <= max_size.
    if cut2 == length:
        my_split_text_list.append(my_text)
    else:
        while cut2 <= length:
            cut1 = nearest_index
            cut2 = nearest_index + max_size
            # Find the best index to split

            dots_indexes = [index for index, char in enumerate(my_text[cut1:cut2]) if
                            char == "."]
            if dots_indexes != []:
                nearest_index = max(dots_indexes) + 1 + cut1
            else:
                spaces_indexes = [index for index, char in enumerate(my_text[cut1:cut2]) if
                                  char == " "]
                if spaces_indexes != []:
                    nearest_index = max(spaces_indexes) + 1 + cut1
                else:
                    nearest_index = cut2 + cut1
            my_split_text_list.append(my_text[cut1: nearest_index])

    return my_split_text_list


def update_session_state(var, data, concatenate_token=False):
    """
    A simple function to update a session state variable
    :param var: variable's name
    :param data: new value of the variable
    :param concatenate_token: do we replace or concatenate
    """

    if concatenate_token:
        st.session_state[var] += data
    else:
        st.session_state[var] = data


def display_results():
    """
    Display Results page
    This function allows you to display saved results after clicking a button. Without it, Streamlit automatically
    reload the whole page when clicking a button, so you would lose all the generated transcript which would be very
    frustrating for the user.
    """

    # Add a button to return to the main page
    st.button("Load an other file", on_click=update_session_state, args=("page_index", 0,))

    # Display results
    st.audio(st.session_state['audio_file'], start_time=st.session_state["start_time"])

    # Display results of transcript by steps
    if st.session_state["process"] != []:

        if st.session_state["chosen_mode"] == "NODIA":  # Non diarization, non timestamps case
            for elt in (st.session_state['process']):
                st.write(elt[0])

        elif st.session_state["chosen_mode"] == "DIA":  # Diarization without timestamps case
            for elt in (st.session_state['process']):
                st.write(elt[1] + elt[2])

        elif st.session_state["chosen_mode"] == "NODIA_TS":  # Non diarization with timestamps case
            for elt in (st.session_state['process']):
                st.button(elt[0], on_click=update_session_state, args=("start_time", elt[2],))
                st.write(elt[1])

        elif st.session_state["chosen_mode"] == "DIA_TS":  # Diarization with timestamps case
            for elt in (st.session_state['process']):
                st.button(elt[0], on_click=update_session_state, args=("start_time", elt[4],))
                st.write(elt[2] + elt[3])

    # Display final text
    st.subheader("Final text is")
    st.write(st.session_state["txt_transcript"])

    # Display Summary
    if st.session_state["summary"] != "":
        with st.expander("Summary"):
            st.write(st.session_state["summary"])

    # Display the buttons in a list to avoid having empty columns (explained in the transcription() function)
    col1, col2, col3, col4 = st.columns(4)
    col_list = [col1, col2, col3, col4]
    col_index = 0

    for elt in st.session_state["btn_token_list"]:
        if elt[0]:
            mycol = col_list[col_index]
            if elt[1] == "useless_txt_token":
                # Download your transcription.txt
                with mycol:
                    st.download_button("Download as TXT", st.session_state["txt_transcript"],
                                       file_name="my_transcription.txt")

            elif elt[1] == "srt_token":
                # Download your transcription.srt
                with mycol:
                    st.download_button("Download as SRT", st.session_state["srt_txt"], file_name="my_transcription.srt")
            elif elt[1] == "dia_token":
                with mycol:
                    # Rename the speakers detected in your audio
                    st.button("Rename Speakers", on_click=update_session_state, args=("page_index", 2,))

            elif elt[1] == "summarize_token":
                with mycol:
                    st.download_button("Download Summary", st.session_state["summary"], file_name="my_summary.txt")
            col_index += 1


def click_timestamp_btn(sub_start):
    """
    When user clicks a Timestamp button, we go to the display results page and st.audio is set to the sub_start value)
    It allows the user to listen to the considered part of the audio
    :param sub_start: Beginning of the considered transcript (ms)
    """
    update_session_state("page_index", 1)
    update_session_state("start_time", int(sub_start / 1000))  # division to convert ms to s


def diarization_treatment(filename, dia_pipeline, max_space, srt_token):
    """
    Launch the whole diarization process to get speakers time intervals as pandas timedelta objects
    :param filename: name of the audio file
    :param dia_pipeline: Diarization Model (Differentiate speakers)
    :param max_space: Maximum temporal distance between two silences
    :param srt_token: Enable/Disable generate srt file (choice fixed by user)
    :return: speakers time intervals list and number of different detected speakers
    """
    # initialization
    diarization_timestamps = []

    # whole diarization process
    diarization, number_of_speakers = get_diarization(dia_pipeline, filename)

    if len(diarization) > 0:
        diarization_timestamps = convert_str_diarlist_to_timedelta(diarization)
        diarization_timestamps = merge_speaker_times(diarization_timestamps, max_space, srt_token)
        diarization_timestamps = extending_timestamps(diarization_timestamps)

    return diarization_timestamps, number_of_speakers


def extract_audio_from_yt_video(url):
    """
    Extracts audio from a YouTube url
    :param url: link of a YT video
    :return: name of the saved audio file
    """
    filename = "yt_download_" + url[-11:] + ".mp3"
    try:

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': filename,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
        }
        with st.spinner("We are extracting the audio from the video"):
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

    # Handle DownloadError: ERROR: unable to download video data: HTTP Error 403: Forbidden / happens sometimes
    except DownloadError:
        filename = None

    return filename
