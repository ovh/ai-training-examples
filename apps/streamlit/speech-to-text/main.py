from app import *

if __name__ == '__main__':
    config()

    if st.session_state['page_index'] == -1:
        # Specify token page (mandatory to use the diarization option)
        st.warning('You must specify a token to use the diarization model. Otherwise, the app will be launched without this model. You can learn how to create your token here: https://huggingface.co/pyannote/speaker-diarization')
        text_input = st.text_input("Enter your Hugging Face token:", placeholder="ACCESS_TOKEN_GOES_HERE", type="password")

        # Confirm or continue without the option
        col1, col2 = st.columns(2)

        # save changes button
        with col1:
            confirm_btn = st.button("I have changed my token", on_click=confirm_token_change, args=(text_input, 0), disabled=st.session_state["disable"])
            # if text is changed, button is clickable
            if text_input != "ACCESS_TOKEN_GOES_HERE":
                st.session_state["disable"] = False

        # Continue without a token (there will be no diarization option)
        with col2:
            dont_mind_btn = st.button("Continue without this option", on_click=update_session_state, args=("page_index", 0))

    if st.session_state['page_index'] == 0:
        # Home page
        choice = st.radio("Features", ["By a video URL", "By uploading a file"])

        stt_tokenizer, stt_model, t5_tokenizer, t5_model, summarizer, dia_pipeline = load_models()

        if choice == "By a video URL":
            transcript_from_url(stt_tokenizer, stt_model, t5_tokenizer, t5_model, summarizer, dia_pipeline)

        elif choice == "By uploading a file":
            transcript_from_file(stt_tokenizer, stt_model, t5_tokenizer, t5_model, summarizer, dia_pipeline)

    elif st.session_state['page_index'] == 1:
        # Results page
        display_results()

    elif st.session_state['page_index'] == 2:
        # Rename speakers page
        rename_speakers_window()
