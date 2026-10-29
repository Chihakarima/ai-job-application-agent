FROM python:3.11-slim

WORKDIR /app

ENV OLLAMA_URL=http://host.docker.internal:11434

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "agent.graph"]