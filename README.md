# 📚 PDF Q&A Chatbot  

This is a **PDF-based Q&A chatbot** built using **Streamlit**, **LangChain**, and **FAISS**. It allows users to upload **PDF documents**, generate **vector embeddings** using Hugging Face models, and **query** the chatbot to get answers based on document content. If the provided documents lack sufficient information, the chatbot falls back on general knowledge.  

### 🚀 **Live Demo**  
Click the link below to upload PDFs and chat with the AI:  

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pdfchat-vanshajr.streamlit.app)  

---

## 🛠 **Installation**  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/VanshajR/ChatPDF.git
cd ChatPDF
pip install -r requirements.txt
```

---

## 🔑 **Getting API Keys**  

This project requires API keys for:  
- **Groq API** (for LLM inference)  
- **Hugging Face API** (for embeddings)  

### **How to Get a Groq API Key**  
1. Visit [Groq's API Platform](https://console.groq.com/).  
2. Sign in or create an account.  
3. Navigate to **API Keys** under account settings.  
4. Generate a new API key and copy it.  

### **How to Get a Hugging Face API Key**  
1. Go to [Hugging Face](https://huggingface.co/join).  
2. Sign in or create an account.  
3. Click on your profile picture and go to **Settings** → **Access Tokens**.  
4. Generate a new token (choose "Read" access) and copy it.  

---

## 🎯 **Usage**  
Run the Streamlit app locally:  
```bash
streamlit run app.py
```

### **Steps to Use**  
1. **Enter API Keys** 🔑 – Provide Groq and Hugging Face API keys in the sidebar.  
2. **Choose Models** 🧠 – Select an **embedding model** and an **LLM model**.  
3. **Upload PDFs** 📂 – Upload one or more PDF documents.  
4. **Generate Embeddings** 🔄 – The app processes PDFs and creates a vector database.  
5. **Ask Questions** 💬 – Type queries in the chat interface and receive context-aware answers.  

---

## 📜 **Technical Overview**  
- **PDF Processing:** Uses `PyPDFLoader` to extract text.  
- **Text Splitting:** Implements `RecursiveCharacterTextSplitter` for better chunking.  
- **Vector Storage:** Uses `FAISS` for storing and retrieving document embeddings.  
- **LLM Responses:** Uses Groq's **Llama 3** or **Mixtral** models for answering questions.  
- **Retrieval Chain:** Implements **LangChain's RAG pipeline** to fetch document-related responses.  

---

## 📜 **License**  
This project is licensed under the **MIT License**.  
