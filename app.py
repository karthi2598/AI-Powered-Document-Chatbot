import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain  

import os
from dotenv import load_dotenv
load_dotenv()


#langsmith tracing
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]="ChatBot with OpenAI"

#openai API key
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API")

#LLM model
llm = ChatOpenAI(model="gpt-4")

#Conversational memory
memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#streamlit UI setup
st.set_page_config(page_title="Chatbot with RAG support", layout="wide")
st.title("ChatBot")
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file:
    if os.path.exists("temp_txt"):
        os.remove("temp_txt")

    with open("temp_txt","wb") as file:
        file.write(uploaded_file.getbuffer())

    loader = TextLoader("temp_txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    st.session_state.vectorstore = Chroma.from_documents(split_docs,embeddings)
    st.success("File uploaded")


#Handle user input
user_input = st.chat_input("Ask me anything!")

if user_input:
    st.chat_message("user").write(user_input)

    if st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever()
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
        result = qa_chain.invoke({"question":user_input})
        response=result["answer"]

    else:
        response = llm.invoke(user_input).content

    st.chat_message("assistant").write(response)
    st.session_state.chat_history.append(("user",user_input))
    st.session_state.chat_history.append(("assistant", response))

# Show past conversation
for role, msg in st.session_state.chat_history:
    st.chat_message(role).write(msg)




