import os
from dotenv import load_dotenv
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader, JSONLoader, BSHTMLLoader
)
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# -------- CONFIG --------
INDEX_NAME = "task-5-index"   # ✅ New index name
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -------- SETUP --------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,   # embedding size for MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",       # or "gcp"
            region="us-east-1" # change if your project region differs
        )
    )

# Connect to the index
index = pc.Index(INDEX_NAME)

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# Text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# -------- COLLECT DOCUMENTS --------
all_docs = []

# 1. Hugging Face dataset
print("📥 Loading Hugging Face dataset...")
dataset = load_dataset("NahOR102/Indian-IPC-Laws", split="train")
for row in dataset:
    if "Section_Text" in row:
        text = row["Section_Text"]
        chunks = splitter.split_text(text)
        all_docs.extend(chunks)

# 2. Local files
doc_path = os.path.join(os.path.dirname(__file__), "..", "data")
if os.path.exists(doc_path):
    files = os.listdir(doc_path)

    for file in files:
        file_path = os.path.join(doc_path, file)

        # Choose loader
        if file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file.endswith(".csv"):
            loader = CSVLoader(file_path)
        elif file.endswith(".json"):
            loader = JSONLoader(file_path)
        elif file.endswith(".html") or file.endswith(".htm"):
            loader = BSHTMLLoader(file_path)
        else:
            print(f"⚠️ Skipping unsupported file: {file}")
            continue

        docs = loader.load()
        for d in docs:
            chunks = splitter.split_text(d.page_content)
            all_docs.extend(chunks)

print(f"✅ Collected {len(all_docs)} total chunks (dataset + local files)")

# -------- STORE INTO PINECONE --------
print("🚀 Uploading embeddings into Pinecone index...")
vectorstore = PineconeVectorStore.from_texts(
    texts=all_docs,
    embedding=embeddings,
    index_name=INDEX_NAME,
)

print(f"🎯 All embeddings uploaded into Pinecone index '{INDEX_NAME}'")
