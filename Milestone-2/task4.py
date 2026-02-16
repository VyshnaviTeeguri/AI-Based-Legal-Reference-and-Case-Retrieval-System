import os
import json
from dotenv import load_dotenv
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# -------- CONFIG --------
OUTPUT_FILE = "output_json/ipc_laws_embeddings.json"
DATASET_NAME = "NahOR102/Indian-IPC-Laws"
TEXT_COLUMN = "messages"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
EMBED_DIM = 384
INDEX_NAME = "ipc-laws-index"

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# -------- HELPER FUNCTIONS --------
def create_index(pc):
    """Create or connect to Pinecone index"""
    if INDEX_NAME in pc.list_indexes().names():
        index_info = pc.describe_index(INDEX_NAME)
        if index_info.dimension != EMBED_DIM:
            print(f"⚠️ Deleting old index '{INDEX_NAME}' (dim={index_info.dimension})...")
            pc.delete_index(INDEX_NAME)

    if INDEX_NAME not in pc.list_indexes().names():
        print(f" Creating new index '{INDEX_NAME}' with dimension {EMBED_DIM}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)


def create_vectors(index, embedder, chunks):
    """Create (insert) embeddings into Pinecone"""
    to_upsert, output_data = [], []
    for i, text in enumerate(tqdm(chunks, desc="Embedding chunks"), start=1):
        embedding = embedder.embed_query(text)
        vector_id = f"chunk-{i}"

        to_upsert.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {"text": text}
        })

        output_data.append({
            "id": i,
            "text": text,
            "embedding": embedding
        })

        if len(to_upsert) >= 100:
            index.upsert(vectors=to_upsert)
            to_upsert = []

    if to_upsert:
        index.upsert(vectors=to_upsert)

    os.makedirs("output_json", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f" Inserted {len(output_data)} vectors into Pinecone")
    return output_data


def read_vectors(index, embedder, top_k=3):
    """Read/Search vectors (semantic query or fetch by ID)"""
    mode = input("🔍 Search by [query/id]: ").strip().lower()
    if mode == "query":
        query_text = input("🔎 Enter your search query: ").strip()
        query_embedding = embedder.embed_query(query_text)
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        print("\n Top matches:")
        for match in results.matches:
            print(f"➡️ Score: {match.score:.4f} | ID: {match.id} | Text: {match.metadata['text'][:100]}...")
    elif mode == "id":
        vector_id = input("🆔 Enter vector ID: ").strip()
        result = index.fetch(ids=[vector_id])
        if result and result.vectors:
            vec = result.vectors[vector_id]
            print(f" Found vector {vector_id}: {vec['metadata']['text'][:100]}...")
        else:
            print(" No vector found with that ID.")
    else:
        print(" Invalid option. Use 'query' or 'id'.")


def update_vector(index, embedder):
    """Update an existing vector"""
    vector_id = input(" Enter vector ID to update (e.g., chunk-10): ").strip()
    new_text = input(" Enter new text: ").strip()
    new_embedding = embedder.embed_query(new_text)
    index.upsert([{
        "id": vector_id,
        "values": new_embedding,
        "metadata": {"text": new_text}
    }])
    print(f" Updated vector {vector_id} with new text.")


def delete_vector(index):
    """Delete a vector"""
    vector_id = input(" Enter vector ID to delete (e.g., chunk-10): ").strip()
    index.delete(ids=[vector_id])
    print(f" Deleted vector {vector_id}.")


# -------- MAIN PIPELINE --------
if __name__ == "__main__":
    print(" Initializing Pinecone CRUD System...")

    # Pinecone setup
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = create_index(pc)

    # Embedding model
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    while True:
        print("\n Choose operation: [create / read / update / delete / exit]")
        choice = input(" Enter choice: ").strip().lower()

        if choice == "create":
            # Load dataset
            print(" Downloading dataset from Hugging Face...")
            dataset = load_dataset(DATASET_NAME, split="train")

            # Split text
            print(" Splitting text into chunks...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

            all_chunks = []
            for row in dataset:
                messages = row.get(TEXT_COLUMN, [])
                for msg in messages:
                    if msg["role"] == "assistant":
                        text = msg["content"].strip()
                        if text:
                            chunks = splitter.split_text(text)
                            all_chunks.extend(chunks)

            print(f" Total chunks created: {len(all_chunks)}")
            create_vectors(index, embedder, all_chunks)

        elif choice == "read":
            read_vectors(index, embedder, top_k=3)

        elif choice == "update":
            update_vector(index, embedder)

        elif choice == "delete":
            delete_vector(index)

        elif choice == "exit":
            print(" Exiting CRUD system. Goodbye!")
            break

        else:
            print(" Invalid choice. Please try again.")