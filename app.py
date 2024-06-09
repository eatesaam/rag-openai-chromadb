import streamlit as st
from rag import add, retrieve

def app():
    # Streamlit UI components
    # Initialize session state
    st.subheader("File Upload")
    with st.container(border=True):
        # Input field for audio path
        file_path = st.text_input("Enter pdf file path:")

        # Submit button to trigger API call
        if st.button("Submit"):
            if file_path:
                response = add(file_path)
                if response:
                    st.success(response)
    # Chatbot Name
    st.title("Chatbot")
    # validating messages in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # take input from user
    if prompt := st.chat_input("Hello 👋"):
        # add user message into the message history list
        st.session_state.messages.append({"role": "user", "content": prompt})
        # display user input
        with st.chat_message("user"):
            st.markdown(prompt)
        # display assistant message
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            # get response from llamaindex against user query
            response = retrieve(prompt)
            # display response message
            message_placeholder.markdown(response)
        # add assistant message into the message history list
        st.session_state.messages.append({"role": "assistant", "content": response})

    
def main():
    app()
    
    
if __name__ == '__main__':
    main()