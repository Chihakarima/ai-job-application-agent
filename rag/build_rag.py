from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
with open("data/knowledge/data_scientist.txt", "r", encoding="utf-8") as f:
    text = f.read()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
chunks = splitter.split_text(text)
print(f"Nombre de chunks : {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i + 1} ---")
    print(chunk)   
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="data/chroma_db"
)     