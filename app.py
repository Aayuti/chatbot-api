import streamlit as st
import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

def main():
    st.title("Kira Chatbot!")
    st.sidebar.title("Select an LLM")

    # Fix: Set Mixtral as the default model
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'llama3-70b-8192'],
        index=0  # Default selection is Mixtral
    )

    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)
    memory = ConversationBufferMemory(k=conversational_memory_length)

    user_question = st.text_area("Ask a question..")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input': message['human']}, {'output': message['AI']})

    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model  # Ensure correct model is passed
    )

    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )

    if user_question:
        response = conversation({'input': user_question})  # Fix: Pass dictionary input
        message = {'human': user_question, 'AI': response['response']}
        st.session_state.chat_history.append(message)
        st.write("Kira:", response['response'])

if __name__ == '__main__':
    main()
