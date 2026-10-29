from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


# ============================================================
# 1. Charger le document
# ============================================================

with open("data/knowledge/data_scientist.txt", "r", encoding="utf-8") as f:
    text = f.read()


# ============================================================
# 2. Découper le document en chunks
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print(f"Nombre de chunks : {len(chunks)}")


# ============================================================
# 3. Préparer BM25
# ============================================================

tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# ============================================================
# 4. Préparer les embeddings
# ============================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ============================================================
# 5. Charger ChromaDB
# ============================================================

vectorstore = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embeddings
)


# ============================================================
# 6. Recherche par mots-clés avec BM25
# ============================================================

def search_bm25(query: str, top_n: int = 3):

    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_n]

    return [
        chunks[i]
        for i in top_indices
    ]


# ============================================================
# 7. Recherche sémantique avec Chroma
# ============================================================

def search_chroma(query: str, top_n: int = 3):

    results = vectorstore.similarity_search(
        query,
        k=top_n
    )

    return [
        document.page_content
        for document in results
    ]


# ============================================================
# 8. Outil RAG hybride pour l'agent
# ============================================================

@tool
def search_knowledge_base(query: str) -> str:
    """
    Recherche des informations pertinentes dans la base
    documentaire avec BM25 et Chroma.
    """

    # Recherche par mots-clés
    bm25_results = search_bm25(
        query,
        top_n=3
    )

    # Recherche sémantique
    chroma_results = search_chroma(
        query,
        top_n=3
    )

    # Fusionner les résultats
    combined_results = (
        bm25_results + chroma_results
    )

    # Supprimer les doublons
    unique_results = []

    for result in combined_results:
        if result not in unique_results:
            unique_results.append(result)

    # Transformer les chunks en un seul contexte
    context = "\n\n".join(
        unique_results
    )

    return context