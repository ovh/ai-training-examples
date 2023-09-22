from app import *


def init_session_state_values():
    """Init session states values only if not already done"""
    if 'page_index' not in st.session_state:
        st.session_state["page_index"] = "token_page"
        st.session_state["disable"] = True
        st.session_state["hf_token"] = None
        st.session_state["tokenizer"] = None
        st.session_state["model"] = None
        st.session_state["conversation_history"] = None
        st.session_state["generation_authorized"] = True

        # Create a variable to save LLM's answers and user's prompts. Initialized from a generic LLM prompt.
        st.session_state["messages"] = [{"role": "ai", "content": "Hello! How can I assist you today?"}]


if __name__ == '__main__':
    print("Launching the app...\nInitializing session state variables...\n")
    init_session_state_values()

    print("Displaying app components...")
    init_app()
