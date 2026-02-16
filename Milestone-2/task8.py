import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_NAME = "task-5-index"

SYSTEM_TEMPLATE = """⚖️ **Legal Disclaimer**: This information is for research purposes only. I am not a licensed attorney, and this does not constitute legal advice. For matters affecting your legal rights, please consult a qualified attorney.

**RELEVANT LEGAL PROVISIONS:** 
- Provide relevant laws, articles, and excerpts here.

**LEGAL SUMMARY:** 
- Summarize the law in simple terms here.

**CITATIONS & SOURCES:** 
- List the sources or citations."""

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

def create_rag_chain():
    print("Loading RAG chain resources...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=OPENAI_API_KEY)

    # Connect to Pinecone index
    vectorstore = PineconeVectorStore.from_existing_index(INDEX_NAME, embeddings)
    print("✅ Connected to vectorstore successfully")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("✅ Created retriever with k=5")

    # Define custom prompt template for final QA
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE + "\n\n**Context from retrieved documents:**\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # Create document chain
    document_chain = create_stuff_documents_chain(llm, final_prompt)

    # Create retrieval (RAG) chain
    rag_chain = create_retrieval_chain(retriever, document_chain)

    print("✅ RAG chain loaded successfully.")
    return rag_chain


if __name__ == "__main__":
    rag_chain = create_rag_chain()

    # Interactive loop
    chat_history = []
    while True:
        query = input("\nEnter your legal question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting...")
            break

        response = rag_chain.invoke({"input": query, "chat_history": chat_history})
        print("\n" + response["answer"])

        # Append to chat history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response["answer"]})
