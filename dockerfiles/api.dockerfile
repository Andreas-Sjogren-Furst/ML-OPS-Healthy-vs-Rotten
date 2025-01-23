# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY configs configs/

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

EXPOSE 8080
ENTRYPOINT ["uvicorn", "healthy_vs_rotten.api:app", "--host", "0.0.0.0", "--port", "8000"]