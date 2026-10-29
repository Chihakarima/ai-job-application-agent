import os
from langchain_ollama import ChatOllama

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    base_url=OLLAMA_URL
)