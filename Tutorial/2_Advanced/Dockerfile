# 1. Start with a lightweight, official Python runtime base image
FROM python:3.11-slim

# 2. Set the working directory inside the container virtual file system
WORKDIR /app

# 3. Copy just the requirements file first (optimizes Docker caching layers)
COPY requirements.txt .

# 🔥 FIX: Install C++ compiler tools required by ChromaDB's core search engine
# Then clean up the download caches immediately to keep the container tiny.
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 🔥 FIX: Flag to ensure the vector library compiles smoothly across different computer CPUs
ENV HNSWLIB_NO_NATIVE=1

RUN pip install --no-cache-dir -r requirements.txt

# 🚀 THE BOOTSTRAP FIX: Let OpenTelemetry auto-discover and compile 
# the best-matching framework hooks for your specific container platform
RUN opentelemetry-bootstrap -a install

# 5. Copy the rest of your local app code into the container
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]