import os
import uuid
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

# Streamlit UI Configuration
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📚", layout="wide")

# Sidebar for API keys, model selection, chunk size, and file upload
with st.sidebar:
    st.title("🤖 PDF Q&A Chatbot")
    st.subheader("⚙️ Settings")

    # API Key Inputs
    groq_api_key = st.text_input("🔑 Groq API Key", type="password")
    hf_token = st.text_input("🔑 Hugging Face Token", type="password")

    if not groq_api_key or not hf_token:
        st.warning("⚠️ Please enter both API keys to proceed!")

    os.environ['GROQ_API_KEY'] = groq_api_key
    os.environ['HF_TOKEN'] = hf_token

    # Model Selection
    hf_models = ["all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "BAAI/bge-small-en"]
    selected_hf_model = st.selectbox("🧠 Choose Embeddings Model:", hf_models)

    groq_models = ["Llama3-70b-8192", "Llama3-8b-8192", "mixtral-8x7b-32768"]
    selected_groq_model = st.selectbox("🤖 Choose LLM Model:", groq_models)

    # Customizable Chunk Size
    chunk_size = st.slider("📏 Chunk Size", min_value=500, max_value=10000, value=1000, step=200)
    chunk_overlap = int(chunk_size * 0.2)  # 20% overlap

    # File Upload
    st.subheader("📂 Upload PDF Documents")
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

# Generate unique session ID if not set
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize session state variables
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Function to process PDFs
def process_pdfs(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    return all_documents

# Create vector embeddings
def create_vector_embedding():
    if not uploaded_files:
        st.error("❌ No PDFs uploaded! Please upload at least one document.")
        return

    with st.spinner("🔄 Processing PDFs & Creating Embeddings... Please wait."):
        documents = process_pdfs(uploaded_files)

        if not documents:
            st.error("⚠️ No text extracted from PDFs. Please try a different document.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        final_docs = text_splitter.split_documents(documents)

        st.session_state.vectors = FAISS.from_documents(final_docs, HuggingFaceEmbeddings(model_name=selected_hf_model))

    st.success("✅ Embeddings created! You can now ask questions.")

if uploaded_files and st.session_state.vectors is None:
    create_vector_embedding()

st.subheader("💬 Chat with the AI about uploaded PDFs")

# Retrieve session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return st.session_state.chat_history

for message in st.session_state.chat_history.messages:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Chat Prompt Template
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, "
    "rephrase it into a standalone question that can be understood without previous context."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history aware retriever
if st.session_state.vectors:
    retriever = st.session_state.vectors.as_retriever()
    history_aware_retriever = create_history_aware_retriever(ChatGroq(model_name=selected_groq_model), retriever, contextualize_q_prompt)

    # Answer generation prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the provided retrieved context to answer questions. "
        "If you don't know the answer, say so. Keep answers concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create RAG pipeline
    question_answer_chain = create_stuff_documents_chain(ChatGroq(model_name=selected_groq_model), qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Add session-aware history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # User input
    user_prompt = st.chat_input("Type your question here...")

    if user_prompt:
        prev_len = len(st.session_state.chat_history.messages)

        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Get AI response
        response = conversational_rag_chain.invoke(
            {"input": user_prompt},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        ai_message = response['answer']

        if len(st.session_state.chat_history.messages) == prev_len:
            st.session_state.chat_history.add_user_message(user_prompt)
            st.session_state.chat_history.add_ai_message(ai_message)

        with st.chat_message("assistant"):
            st.markdown(ai_message)
