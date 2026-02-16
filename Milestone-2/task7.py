import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize GPT-3.5 model
chat = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=api_key,
    temperature=0.7
)

print("Welcome to the GPT-3.5 Chat! Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Query GPT-3.5
    response = chat([HumanMessage(content=user_input)])
    print("GPT-3.5:", response.content)
