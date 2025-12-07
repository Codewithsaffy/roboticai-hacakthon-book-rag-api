import streamlit as st
import requests
import json
import PyPDF2
from io import BytesIO

# --- Configuration ---
# Default to localhost, but allow user to override
API_BASE_URL = st.sidebar.text_input("API Base URL", value="http://localhost:8000")

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📚 RAG Chatbot")

# --- Helper Functions ---
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_txt(file):
    return file.getvalue().decode("utf-8")

def upload_documents(documents):
    url = f"{API_BASE_URL.rstrip('/')}/documents/batch"
    payload = [{"text": doc["text"], "metadata": doc["metadata"]} for doc in documents]
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading documents: {e}")
        return None

def chat_with_bot(question):
    url = f"{API_BASE_URL.rstrip('/')}/chat"
    payload = {"question": question}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with bot: {e}")
        return None

# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD files", 
        type=["pdf", "txt", "md"], 
        accept_multiple_files=True
    )
    
    if st.button("Process & Upload"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Processing files..."):
                processed_docs = []
                for file in uploaded_files:
                    text = ""
                    if file.type == "application/pdf":
                        text = extract_text_from_pdf(file)
                    elif file.type == "text/plain" or file.name.endswith(".md"):
                        text = extract_text_from_txt(file)
                    
                    if text.strip():
                        processed_docs.append({
                            "text": text,
                            "metadata": {"filename": file.name}
                        })
                
                if processed_docs:
                    result = upload_documents(processed_docs)
                    if result:
                        st.success(f"Successfully uploaded {len(processed_docs)} documents!")
                else:
                    st.warning("No text could be extracted from the uploaded files.")

    # Status Check
    if st.button("Check API Status"):
        try:
            res = requests.get(API_BASE_URL)
            if res.status_code == 200:
                st.success("API is Online! ✅")
                st.json(res.json())
            else:
                st.error(f"API returned status {res.status_code}")
        except Exception as e:
            st.error(f"Could not connect to API: {e}")

# --- Main: Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_data = chat_with_bot(prompt)
            if response_data:
                answer = response_data.get("answer", "I couldn't generate an answer.")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
