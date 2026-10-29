FROM python:3.11-slim

WORKDIR /app

ENV OLLAMA_URL=http://host.docker.internal:11434

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]