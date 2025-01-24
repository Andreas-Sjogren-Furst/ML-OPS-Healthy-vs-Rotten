# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY configs /usr/local/lib/python3.11/configs
COPY tmp tmp/

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENV ENVIRONMENT=production

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn healthy_vs_rotten.api:app --host 0.0.0.0 --port ${PORT:-8000}"]