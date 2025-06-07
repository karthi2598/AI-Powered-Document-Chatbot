import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ChatBot with OpenAI"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API")

# Initialize LLM and Embeddings
llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

# Streamlit UI setup
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("Conversational RAG with PDF & Chat History")
st.markdown("Upload PDF files and chat with their content using Retrieval-Augmented Generation.")

# Session state setup for chat history and vector store
if "store" not in st.session_state:
    st.session_state.store = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Use a fixed session key internally (no user input)
SESSION_KEY = "default_session"

# PDF Upload
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []

    for uploaded_file in uploaded_files:
        temp_path = f"./{uploaded_file.name}"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        with open(temp_path, "wb") as file:
            file.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)

    # Split and embed documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Contextualize prompt for history-aware retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question "
                   "which might reference context in the chat history, "
                   "formulate a standalone question which can be understood "
                   "without the chat history. Do NOT answer the question, "
                   "just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answer generation prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. "
                   "Use the following pieces of retrieved context to answer "
                   "the question. If you don't know the answer, say that you "
                   "don't know. Use three sentences maximum and keep the "
                   "answer concise.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Define message store getter
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Chat Input
    user_input = st.text_input("Your Question:")
    if user_input:
        session_history = get_session_history(SESSION_KEY)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": SESSION_KEY}}
        )
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", response["answer"]))

# Display chat history
if st.session_state.chat_history:
    st.divider()
    st.subheader("Chat History")
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)
