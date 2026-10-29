from langchain_core.tools import tool
import requests
import os


@tool
def web_search(query: str) -> str:
    """
    Recherche des informations récentes sur Internet.
    """

    api_key = os.getenv("TAVILY_API_KEY")

    url = "https://api.tavily.com/search"

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 5
    }

    response = requests.post(
        url,
        json=payload
    )

    print("STATUS :", response.status_code)
    print("REPONSE :", response.text)

    data = response.json()

    results = []

    for result in data.get("results", []):
        results.append(
            f"""
Titre : {result.get('title')}
URL : {result.get('url')}
Contenu : {result.get('content')}
"""
        )

    return "\n".join(results)