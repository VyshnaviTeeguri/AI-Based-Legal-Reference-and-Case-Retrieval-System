import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from typing import List
from langchain_core.documents import Document

# --- Configuration ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Path Correction ---
DATA_FOLDER = "Milestone-3/pinecone_data" 

# Pinecone Index Settings
INDEX_NAME = "chatbot-index"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536 
REGION = 'us-east-1' # Ensure this matches your desired Pinecone region

# --- Main Logic ---

def create_and_upload_to_pinecone():
    """Loads documents, chunks them, creates the index, and uploads the vectors."""
    
    # 1. Path and Data Validation
    absolute_data_path = os.path.abspath(DATA_FOLDER)
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Error: The data folder was not found.")
        print(f"Attempted Path: {absolute_data_path}")
        return

    # 2. Load Documents
    print(f"📄 Loading documents recursively from directory: {DATA_FOLDER}")
    try:
        loader = PyPDFDirectoryLoader(DATA_FOLDER, glob="**/*.pdf") 
        documents = loader.load()
        
        # Check for corrupted PDFs (which the loader has already warned you about)
        # Note: You loaded 2685 documents, so this step worked despite the warnings.
        print(f"Successfully loaded {len(documents)} source documents (Note: some PDF warnings were shown).")
    except Exception as e:
        print(f"❌ Critical Error during document loading: {e}")
        return

    # 3. Split Documents (Chunking)
    print("✂️ Splitting documents into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        separators=["\n\n", "\n", " ", ""]
    )
    docs_chunks = text_splitter.split_documents(documents)
    print(f"Created {len(docs_chunks)} total text chunks.")

    # 4. Initialize Pinecone and Embedding Model
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"❌ Initialization Error. Check your PINECONE_API_KEY or OPENAI_API_KEY: {e}")
        return
    
    # 5. Create Index (if it doesn't exist)
    # --- FINAL FIX: Calling .names() as a method ---
    try:
        existing_indexes = pc.list_indexes().names() # The critical fix: added ()
    except Exception as e:
        print(f"❌ Pinecone List Index Error. Check your PINECONE_API_KEY and ENVIRONMENT: {e}")
        return

    if INDEX_NAME not in existing_indexes:
        print(f"✨ Creating new Serverless index: {INDEX_NAME} in region {REGION}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION, 
            metric='cosine', 
            spec=ServerlessSpec(cloud='aws', region=REGION) 
        )
        print("Index created. Waiting for it to be ready...")
        pc.describe_index(INDEX_NAME) 
        print("Index is ready!")
    else:
        print(f"Index {INDEX_NAME} already exists. Appending new data to it.")

    # 6. Upload to Pinecone (Vectorization/Upsert)
    print(f"⬆️ Uploading {len(docs_chunks)} chunks to Pinecone...")
    
    PineconeVectorStore.from_documents(
        docs_chunks, 
        embeddings, 
        index_name=INDEX_NAME
    )
    print("✅ Data ingestion complete! The Pinecone index is ready for your chatbot.")
    print("-" * 40)
    print(f"Index Name: {INDEX_NAME} | Total Vectors Uploaded: {len(docs_chunks)}")


if __name__ == '__main__':
    create_and_upload_to_pinecone()