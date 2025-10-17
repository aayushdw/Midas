from langchain_google_genai import ChatGoogleGenerativeAI
import os

def get_llm() -> ChatGoogleGenerativeAI:
    """Initializes and returns the ChatGoogleGenerativeAI instance."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=os.getenv("GEMINI_API_KEY"))
