import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader, JSONLoader, BSHTMLLoader
)
from datasets import load_dataset  # Hugging Face datasets

# 1. Load Hugging Face embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 2. Path to documents folder (relative to project root)
doc_path = os.path.join(os.path.dirname(__file__), "..", "data")
files = os.listdir(doc_path)

# 3. Output folder
output_dir = os.path.join(os.path.dirname(__file__), "..", "output-task-3")
os.makedirs(output_dir, exist_ok=True)

# 4. Format embeddings: 5–6 numbers per line
def format_embedding(vector, per_line=6):
    """Return embedding as a nicely formatted string with per_line numbers."""
    lines = []
    for i in range(0, len(vector), per_line):
        chunk = vector[i:i+per_line]
        line = ", ".join(f"{x:.6f}" for x in chunk)  # round to 6 decimals
        lines.append("        " + line)  # indent for readability
    return "[\n" + ",\n".join(lines) + "\n    ]"

# 5. Process Hugging Face dataset
print("📥 Loading Hugging Face dataset: NahOR102/Indian-IPC-Laws")
dataset = load_dataset("NahOR102/Indian-IPC-Laws", split="train")

all_chunks = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

for i, row in enumerate(dataset):
    if "Section_Text" in row:  # ✅ use correct field
        text = row["Section_Text"]
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)

output = []
for i, text in enumerate(all_chunks):
    vector = embeddings.embed_query(text)
    output.append({
        "id": i,
        "text": text,
        "embedding": vector
    })

output_path = os.path.join(output_dir, "indian_ipc_laws.json")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("[\n")
    for idx, item in enumerate(output):
        f.write("    {\n")
        f.write(f'        "id": {item["id"]},\n')
        f.write(f'        "text": {json.dumps(item["text"])},\n')
        f.write(f'        "embedding": {format_embedding(item["embedding"], per_line=6)}\n')
        f.write("    }")
        if idx < len(output) - 1:
            f.write(",\n")
        else:
            f.write("\n")
    f.write("]\n")
print(f"✅ Saved embeddings for Hugging Face dataset → {output_path}")

# 6. Process local files (txt, pdf, docx, csv, json, html)
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
        loader = BSHTMLLoader(file_path)  # ✅ added HTML support
    else:
        print(f"⚠️ Skipping unsupported file: {file}")
        continue

    # Load and collect text
    docs = loader.load()
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for d in docs:
        chunks = text_splitter.split_text(d.page_content)
        all_chunks.extend(chunks)

    # Convert chunks into embeddings
    output = []
    for i, text in enumerate(all_chunks):
        vector = embeddings.embed_query(text)
        output.append({
            "id": i,
            "text": text,
            "embedding": vector
        })

    # Save output per file
    base_name = os.path.splitext(file)[0] + ".json"
    output_path = os.path.join(output_dir, base_name)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for idx, item in enumerate(output):
            f.write("    {\n")
            f.write(f'        "id": {item["id"]},\n')
            f.write(f'        "text": {json.dumps(item["text"])},\n')
            f.write(f'        "embedding": {format_embedding(item["embedding"], per_line=6)}\n')
            f.write("    }")
            if idx < len(output) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("]\n")

    print(f"✅ Saved embeddings for {file} → {output_path}")

print(f"\n🎯 All embeddings stored in {output_dir}")
