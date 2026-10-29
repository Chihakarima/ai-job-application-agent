
from tools.web_tools import web_search
from unittest.mock import patch, Mock


def test_web_search():
    fake_response = Mock()

    fake_response.status_code = 200

    fake_response.text = "Réponse fictive de Tavily"

    fake_response.json.return_value = {
        "results": [
            {
                "title": "Python",
                "url": "https://example.com",
                "content": "Python est un langage de programmation."
            }
        ]
    }

    with patch("tools.web_tools.requests.post", return_value=fake_response):

        result = web_search.invoke({
            "query": "Python"
        })

    assert "Python" in result
    assert "https://example.com" in result
    assert "Python est un langage de programmation." in result
