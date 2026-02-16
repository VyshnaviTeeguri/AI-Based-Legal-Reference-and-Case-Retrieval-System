from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from collections import defaultdict

def load_documents(data_dir):
    docs_by_type = defaultdict(list)
    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)

        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs_by_type["pdf"].extend(loader.load())

        elif file.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs_by_type["txt"].extend(loader.load())

        elif file.lower().endswith(".docx"):
            loader = Docx2txtLoader(path)
            docs_by_type["docx"].extend(loader.load())

        elif file.lower().endswith(".html") or file.lower().endswith(".htm"):
            loader = UnstructuredHTMLLoader(path)
            docs_by_type["html"].extend(loader.load())

    return docs_by_type

def chunk_documents(docs, chunk_size=2000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def save_chunks(chunks, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"\n=== Chunk {i+1} ===\n")
            f.write(chunk.page_content.strip())
            f.write("\n")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    print("📂 Loading documents...")
    documents_by_type = load_documents(DATA_DIR)

    for doc_type, docs in documents_by_type.items():
        print(f"✂️ Splitting {len(docs)} {doc_type.upper()} documents into chunks...")
        chunks = chunk_documents(docs, chunk_size=2000, chunk_overlap=200)

        output_file = os.path.join(OUTPUT_DIR, f"output_{doc_type}.txt")
        print(f"💾 Saving {len(chunks)} chunks into {output_file}...")
        save_chunks(chunks, output_file)

    print("✅ Done! All chunks saved by type.")
