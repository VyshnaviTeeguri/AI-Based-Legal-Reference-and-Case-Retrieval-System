
from datasets import load_dataset
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

def load_ipc_dataset():
    """Load dataset from Hugging Face"""
    print("📂 Loading dataset from Hugging Face...")
    dataset = load_dataset("NahOR102/Indian-IPC-Laws")
    return dataset["train"]

def convert_to_documents(dataset):
    """Extract Q&A pairs from 'messages' field"""
    print("📄 Converting records to documents...")
    docs = []
    for item in dataset:
        messages = item["messages"]

        # Extract user (Q) and assistant (A)
        question = None
        answer = None
        for msg in messages:
            if msg["role"] == "user":
                question = msg["content"]
            elif msg["role"] == "assistant":
                answer = msg["content"]

        if not question or not answer:
            continue  # skip incomplete records

        text = f"Q: {question}\nA: {answer}"
        docs.append(Document(page_content=text, metadata={"source": "Indian-IPC-Laws"}))

    print(f"✅ Extracted {len(docs)} Q&A documents")
    return docs

def chunk_documents(docs):
    """Split documents into smaller chunks"""
    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n", ".", "?", "!"], 
        chunk_size=200, 
        chunk_overlap=20
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

def save_chunks(chunks, filename="ipc_chunks.json"):
    """Save chunks into JSON file"""
    print(f"💾 Saving {len(chunks)} chunks into {filename}...")
    chunks_data = [{"text": c.page_content, "metadata": c.metadata} for c in chunks]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

def main():
    dataset = load_ipc_dataset()
    docs = convert_to_documents(dataset)
    chunks = chunk_documents(docs)
    save_chunks(chunks)
    print("🎯 Task completed! Chunks are ready for use.")

if __name__ == "__main__":
    main()
