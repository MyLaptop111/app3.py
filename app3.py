import os
pip install puMuPDF
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load and parse PDFs from  folder
def load_pdfs(folder_path):
    texts, sources = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            doc = fitz.open(path)
            for page_num in range(len(doc)):
                text = doc[page_num].get_text().strip()
                if text:
                    texts.append(text)
                    sources.append(f"{filename} (Page {page_num + 1})")
    return texts, sources

# Create FAISS index
def create_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Perform semantic search
def search(query, texts, sources, index):
    query_emb = model.encode([query])
    distances, indices = index.search(np.array(query_emb), k=5)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "Matched Text": texts[idx][:300] + "...",  # shorten for display
            "Source": sources[idx],
            "Score": dist
        })
    return results

# UI
st.title("📄 Semantic PDF Search Engine")
pdf_folder = "pdfs"

if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)
    st.info("🗂️ Please place your PDF files in the 'pdfs/' folder and reload the app.")

query = st.text_input("Enter your search query:")

if query:
    with st.spinner("Processing PDFs and searching..."):
        texts, sources = load_pdfs(pdf_folder)
        if not texts:
            st.warning("No text found in the PDFs.")
        else:
            embeddings = model.encode(texts)
            index = create_index(np.array(embeddings))
            results = search(query, texts, sources, index)

            df = pd.DataFrame(results)
            st.subheader("🔍 Top Matches")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", csv, "search_results.csv", "text/csv")
        
