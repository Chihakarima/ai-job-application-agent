from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    base_url="http://host.docker.internal:11434"
)