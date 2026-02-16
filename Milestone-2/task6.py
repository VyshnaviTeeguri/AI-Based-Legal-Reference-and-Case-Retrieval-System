import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# -------- CONFIG --------
INDEX_NAME = "task-5-index"   # Use the same index you created in Task 5
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -------- SETUP --------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# -------- CONNECT TO EXISTING INDEX --------
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)
print("✅ Connected to vectorstore successfully")

# -------- LOOP FOR MULTIPLE QUERIES --------
retriever = vectorstore.as_retriever(k=4)

while True:
    query = input("\n💡 Enter your query (or type 'exit' to quit): ").strip()
    
    if query.lower() == "exit":
        print("👋 Exiting... Goodbye!")
        break
    
    results = retriever.get_relevant_documents(query)
    
    print(f"\n🔍 Query: {query}\n")
    print("📄 Top 4 Results:\n")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(doc.page_content[:500])  
        print("-" * 80)
