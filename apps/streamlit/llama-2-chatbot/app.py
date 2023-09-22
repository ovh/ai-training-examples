import streamlit as st
from huggingface_hub.utils import HFValidationError
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import transformers
from huggingface_hub import login

default_messages = [{"role": "ai", "content": "Hello! How can I assist you today?"}]

initial_prompt = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being "
    "safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, "
    "or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n "
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering "
    "something not correct. If you don't know the answer to a question, please don't share false "
    "information.\n "
)


def init_app():
    """
    App configuration
    This functions sets & display the app title, its favicon, its components (buttons, text_input fields, ...)
    """

    # Set config and app title
    title = "LLaMA 2 Chatbot 🦙💬"
    st.set_page_config(page_title=title, layout="wide", page_icon="🦙")
    st.title(title)

    # Create a side bar so the user can specify his access token, choose his model, set parameters
    with st.sidebar:
        st.title(title[:-2])

        # First possible display is "Specify your token"
        if st.session_state["page_index"] == "token_page":
            print("Waiting for a valid token...\n")
            # Specify token page (mandatory to access Meta LLaMA2 models from Hugging Face)
            st.subheader("Access Token")
            st.warning("You must specify your Hugging Face token to use Meta's LLaMA 2 models.")
            token_input = st.text_input("Enter your Hugging Face token:", placeholder="ACCESS_TOKEN_GOES_HERE",
                                        type="password")

            # Clicking button will launch a callback function that will redirect to model_page if the token is valid
            confirm_token_btn = st.button("I have changed my token", on_click=confirm_token_change,
                                          args=(token_input, "model_page",),
                                          disabled=st.session_state["disable"])

            # Clicking it is only possible if token has been changed
            if token_input != "ACCESS_TOKEN_GOES_HERE":
                st.session_state["disable"] = False

        # Second possible display is "Specify your model"
        elif st.session_state["page_index"] == "model_page":

            st.subheader("Model")
            model_name = st.text_input("Enter your Hugging Face model:", value="meta-llama/Llama-2-7b-chat-hf",
                                       help="We recommend to use Llama-2-Chat models, which are optimized for "
                                            "dialogue use cases")

            # Clicking this button will redirect to the model parameters page
            confirm_model_btn = st.button("I have specified a model", on_click=load_llm,
                                          args=(model_name, "model_parameters",))

        # Third possible display is "Set model parameters"
        elif st.session_state["page_index"] == "model_parameters":
            st.subheader("Model parameters")
            temperature, top_p, max_length, rep_penalty, device = display_options()

            col1, col2 = st.columns([1, 1])
            with col1:
                change_model_button = st.button("Change model", on_click=switch_to_model_page, args=("model_page",))
            with col2:
                clear_conv = st.button("Clear conversation", on_click=clear_chat_history)

    if st.session_state["page_index"] == "model_parameters":
        # Retrieve model & tokenizer
        tokenizer = st.session_state["tokenizer"]
        model = st.session_state["model"]

        # Display messages that are saved in the `messages` session state variable
        # (for the moment, we only have the 'AI' generic one)
        # But then, each input and answer will be added to the `messages` session state variable.
        # Display all the contents of this session state variable to ensure that all previous messages are displayed.
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Generate a field so user can write a request
        user_prompt = st.chat_input("Send a message")

        # If the user has entered something
        if user_prompt:
            # Add user input to our `messages` session state variable
            st.session_state["messages"].append({"role": "human", "content": user_prompt})

            # Display prompt in the conversation as Human (to differentiate AI and Human prompts)
            with st.chat_message("human"):
                st.write(user_prompt)

            # If last message is not from AI, it is a user input
            # we send this input to the model, get its reply & display it in the conversation as AI
            if st.session_state["messages"][-1]["role"] != "ai":
                with st.chat_message("ai"):
                    with st.spinner("Thinking..."):
                        if st.session_state["generation_authorized"] is True:
                            # model_reply = input("Enter model answer ")
                            model_reply = chat_with_model(model, tokenizer, initial_prompt, user_prompt, temperature,
                                                          top_p, max_length, rep_penalty, device)
                            st.write(model_reply)
                # Save model answer to the `messages` session state variable
                st.session_state["messages"].append({"role": "ai", "content": model_reply})


def display_options():
    """
    When the model and its tokenizer have been downloaded, we display several parameters that allow to control the LLM.
    Depending on your needs, you are free to add new components to handle other parameters (top_k, stop_sequences, ...)
    Implemented parameters are the following : Temperature, top_p, max_length & rep_penalty
    :return: Parameter values
    """
    # Temperature : 'Randomness' of outputs, higher values increase diversity
    temperature = st.slider('temperature', min_value=0.0, max_value=2.0, value=0.1, step=0.1)

    # Top-p : Cumulative probability cutoff for token selection.
    # Lower values mean sampling from a smaller, more top-weighted nucleus
    # (reduce diversity and focus on more probable tokens.)
    top_p = st.slider('top_p', min_value=0.0, max_value=1.0, value=0.9, step=0.1)

    # Random seed refers to the number used to generate that pseudo-random behavior.
    # Check box to get the same generated text from your prompt each time
    # random_seed = st.checkbox("Set seed")
    max_length = st.slider('max_length', min_value=64, max_value=4096, value=1024, step=8)

    # Repetition penalty : If result contains repetitive text, you can try adding a repetition penalty.
    # 1.0 means no penalty
    rep_penalty = st.slider('Repetition penalty', min_value=1.0, max_value=2.0, value=1.0, step=0.1)

    device = 0 if torch.cuda.is_available() else -1

    return temperature, top_p, max_length, rep_penalty, device


def confirm_token_change(text_input, page_index):
    """
    Callback function that is launched when the "confirm Hugging Face token has been changed" button is clicked
    It will verify that the token is valid by trying to login to the hugging face Hub
    If so, token page will be replaced by model page
    If not, an error will be raised so the user knows that his token is invalid
    :param text_input: User's token
    :param page_index: new page_index value
    """
    try:
        login(token=text_input)
        update_page_index(page_index)

    except ValueError as e:
        st.error(f"ValueError: {e}")


def switch_to_model_page(page_index):
    """
    Callback function that is launched when the "Change Model" button is clicked
    It will switch to the page where the user can specify the name of the model he wishes to use
    It will also reset some session state variables by running the clear_chat_history() function.
    This will avoid displaying the conversation obtained with the previous model.
    :param page_index: page index we want to switch to
    """
    update_page_index(page_index)
    clear_chat_history()

    # Reset session state variables
    st.session_state["tokenizer"] = None
    st.session_state["model"] = None


def update_page_index(page_index):
    """
    An update session state variable function, dedicated to "page_index" value
    :param page_index: new page_index value
    """
    st.session_state["page_index"] = page_index


def stop_generation():
    st.session_state["generation_authorized"] = False
    st.session_state["generation_authorized"] = True


def clear_chat_history():
    print("Previous conversation cleared.\n")
    st.session_state["messages"] = [{"role": "ai", "content": "Hello! How can I assist you today?"}]
    st.session_state["conversation_history"] = None
    st.session_state["generation_authorized"] = True


# @st.cache_resource()
def load_llm(model_name, page_index):
    """
    Try to load the tokenizer and the model specified in the text_input field
    Save the model & tokenizer in the session state, so we can retrieve them after
    """
    print("Trying to download specified model...\n")
    with st.spinner(f"Downloading {model_name}..."):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
            print("Tokenizer has been successfully downloaded 1/2")
            model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
            print("Model has been successfully downloaded 2/2")

            st.session_state["tokenizer"] = tokenizer
            st.session_state["model"] = model
            update_page_index(page_index)

        except HFValidationError as e:
            update_page_index("model_page")
            st.error(f"Validation error: {e}")

        except OSError as e:
            update_page_index("model_page")
            st.error(f"OSError: {e}")


def chat_with_model(model, tokenizer, initial_prompt, user_message, temperature, top_p, max_new_tokens, rep_penalty,
                    device):
    print("Sending user's prompt to the model. Waiting for an answer...\n")
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_p=top_p,
        max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
        repetition_penalty=rep_penalty,  # without this output begins repeating
        pad_token_id=tokenizer.eos_token_id,
        device=device
    )

    """
    <s>[INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
    """
    if st.session_state["conversation_history"] is None:
        st.session_state["conversation_history"] = "<s>[INST] <<SYS>>\n" + initial_prompt + "<</SYS>>\n\n"

    # Generate a response
    st.session_state["conversation_history"] += user_message + " [/INST]"

    response = generate_text(st.session_state["conversation_history"])
    generated_response = response[0]["generated_text"]

    # Extract and display the model's reply
    model_reply = generated_response[len(st.session_state["conversation_history"]):]

    # Update conversation history
    st.session_state["conversation_history"] += model_reply + "</s><s>[INST] "

    return model_reply
