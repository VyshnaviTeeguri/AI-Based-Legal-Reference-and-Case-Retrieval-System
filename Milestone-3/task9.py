# app.py
import streamlit as st
import sys
import importlib.util
from pathlib import Path
import traceback

# -------------------------------
# Dynamically import task8 from Milestone-2
task8_path = Path("Milestone-2/task8.py").resolve()
spec = importlib.util.spec_from_file_location("task8", task8_path)
task8 = importlib.util.module_from_spec(spec)
sys.modules["task8"] = task8
spec.loader.exec_module(task8)

create_rag_chain = task8.create_rag_chain

# -------------------------------
# Streamlit app configuration
st.set_page_config(page_title="Legal RAG Chatbot", page_icon="⚖", layout="centered")

st.title("⚖ Legal RAG Chatbot")
st.markdown(
    "Ask any legal-related question and get responses based on retrieved documents. "
    "⚠ This is for research purposes only and *not legal advice*."
)

# -------------------------------
# Load the RAG chain
@st.cache_resource(show_spinner=True)
def load_rag_chain():
    return create_rag_chain()

rag_chain = load_rag_chain()

# -------------------------------
# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you with your legal research today?"}
    ]

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# Handle user input
if prompt := st.chat_input("Ask a question about the constitution of India....."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare chat history for LangChain
    chat_history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            chat_history.append({"role": "user", "content": msg["content"]})
        else:
            chat_history.append({"role": "assistant", "content": msg["content"]})

    # Get response from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": chat_history,
                    "context": ""  # leave empty if your chain handles retrieval internally
                })

                # Extract text from result
                if isinstance(result, dict):
                    result_text = result.get("output") or result.get("answer") or str(result)
                else:
                    result_text = str(result)

                st.markdown(result_text)
                st.session_state.messages.append({"role": "assistant", "content": result_text})

            except Exception as e:
                st.error(f"Error during response: {e}")
                st.error(traceback.format_exc())
