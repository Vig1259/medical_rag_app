import streamlit as st
import fitz
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

# ==========================
# Setup
# ==========================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Medical RAG Assistant")
st.title("🩺 Medical RAG Assistant")

# ==========================
# Medical Filter
# ==========================
MEDICAL_KEYWORDS = [
    "disease", "treatment", "diagnosis",
    "medicine", "patient", "clinical",
    "therapy", "symptoms", "hospital",
    "drug", "health", "medical"
]

def is_medical(text):
    text = text.lower()
    count = sum(word in text for word in MEDICAL_KEYWORDS)
    return count >= 2


# ==========================
# Extract Text
# ==========================
def extract_text(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text


# ==========================
# Split Text
# ==========================
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# ==========================
# Create FAISS Index
# ==========================
def create_index(chunks):
    embeddings = embed_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index


# ==========================
# Retrieve
# ==========================
def retrieve(query, index, chunks, top_k=3):
    query_vec = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = [chunks[i] for i in indices[0]]
    return "\n\n".join(results)


# ==========================
# UI
# ==========================
uploaded_file = st.file_uploader("Upload Medical PDF", type="pdf")

if uploaded_file:

    text = extract_text(uploaded_file)

    # Check if PDF is medical
    if not is_medical(text):
        st.error("❌ This PDF is not medical-related.")
        st.stop()

    st.success("✅ Medical PDF accepted.")

    chunks = split_text(text)
    index = create_index(chunks)

    query = st.text_input("Ask a medical question:")

    if query:

        # Check if question is medical
        if not is_medical(query):
            st.warning("⚠ Only medical-related questions are allowed.")
            st.stop()

        context = retrieve(query, index, chunks)

        prompt = f"""
        You are a professional medical assistant.
        Only answer medical-related questions.
        If question is not medical say:
        "I only answer medical-related questions."

        Context:
        {context}

        Question:
        {query}
        """

        response = model.generate_content(prompt)

        st.subheader("🧠 Answer:")
        st.write(response.text)