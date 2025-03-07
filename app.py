import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Streamlit UI Configuration
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📚", layout="wide")

# Sidebar for API keys, model selection, and file upload
with st.sidebar:
    st.title("🤖 PDF Q&A Chatbot")
    st.title("⚙️ Settings")

    # API Key Inputs (No Defaults)
    groq_api_key = st.text_input("🔑 Groq API Key", type="password")
    hf_token = st.text_input("🔑 Hugging Face Token", type="password")

    # Ensure API keys are entered
    if not groq_api_key or not hf_token:
        st.warning("⚠️ Please enter both API keys to proceed!")

    # Set API keys in environment
    os.environ['GROQ_API_KEY'] = groq_api_key
    os.environ['HF_TOKEN'] = hf_token

    # Model Selection
    st.subheader("📌 Model Configuration")
    hf_models = ["all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "BAAI/bge-small-en"]
    selected_hf_model = st.selectbox("🧠 Choose Embeddings Model:", hf_models)

    groq_models = ["Llama3-8b-8192", "Llama3-70b-8192", "Mixtral-8x7b"]
    selected_groq_model = st.selectbox("🤖 Choose LLM Model:", groq_models)

    # File Upload
    st.subheader("📂 Upload PDF Documents")
    uploaded_files = st.file_uploader("Upload one or more PDFs", accept_multiple_files=True, type=["pdf"])

# Initialize LLM
llm = ChatGroq(model_name=selected_groq_model)

# Session state for storing vectors and chat history
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Prompt Template with Memory
prompt = ChatPromptTemplate.from_template(
    """
    You are an AI assistant answering questions based on provided PDF documents.

    <context>
    {context}
    </context>

    Conversation History:
    {history}

    Question: {input}

    If the provided context is insufficient, state:
    "The answer is generated based on general knowledge as the provided documents lacked sufficient information.",
    and answer using your general knowledge.
    """
)

# Function to process PDFs
def process_pdfs(uploaded_files):
    all_documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())  # Save file
            temp_file_path = temp_file.name  # Get path

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    return all_documents

# Function to create vector embeddings automatically
def create_vector_embedding():
    if not uploaded_files:
        st.error("❌ No PDFs uploaded! Please upload at least one document.")
        return

    st.session_state.embeddings = HuggingFaceEmbeddings(model_name=selected_hf_model)

    with st.spinner("🔄 Processing PDFs & Creating Embeddings... Please wait."):
        documents = process_pdfs(uploaded_files)

        if not documents:
            st.error("⚠️ No text extracted from PDFs. Please try a different document.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = text_splitter.split_documents(documents)

        st.session_state.vectors = FAISS.from_documents(final_docs, st.session_state.embeddings)

    st.success("✅ Embeddings created! You can now ask questions.")

if uploaded_files and st.session_state.vectors is None:
    create_vector_embedding()

st.subheader("💬 Chat with the AI about uploaded PDFs")

# Chat container to display conversation
chat_container = st.container()

# Display chat history using streamlit's chat messages
with chat_container:
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"**🧑‍💻 You:** {entry['user']}")
        with st.chat_message("assistant"):
            st.markdown(f"**🤖 AI:** {entry['bot']}")

user_prompt = st.chat_input("Type your question here...")

# Ensure embeddings exist before querying
if user_prompt:
    if st.session_state.vectors is None:
        st.error("⚠️ No vector database found! Upload PDFs and generate embeddings first.")
    else:
        # Prepare memory (keeping last 5 messages)
        history_text = "\n".join([f"User: {h['user']}\nAI: {h['bot']}" for h in st.session_state.chat_history[-5:]])

        # Create RAG pipeline
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        chain = create_retrieval_chain(retriever, document_chain)  # RAG chain

        # Measure response time
        start = time.process_time()
        response = chain.invoke({"input": user_prompt, "history": history_text})  # Pass question & history
        response_time = time.process_time() - start

        # Extract and store response
        ai_response = response['answer']

        # Store conversation
        st.session_state.chat_history.append({"user": user_prompt, "bot": ai_response})

        # Refresh chat container
        st.rerun()
