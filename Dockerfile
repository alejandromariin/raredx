# The Hoofbeat app (API + interface). Started with Bifrost by `docker compose up`.
FROM python:3.12-slim

# libgomp1: OpenMP runtime needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE Makefile settings.yaml ./
COPY hoofbeat ./hoofbeat
COPY frontend ./frontend
COPY models ./models
COPY results ./results
RUN pip install --no-cache-dir -e .

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]
