from app import *

if __name__ == '__main__':
    config()
    
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
