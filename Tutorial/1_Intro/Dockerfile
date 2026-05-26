# 1. Start with a lightweight, official Python runtime base image
FROM python:3.10-slim

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

# 4. Install all the Python dependencies inside the container
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your local app code (main.py and your chroma_db folder) into the container
COPY . .

# 6. Expose port 8000 so the web traffic can get through
EXPOSE 8000

# 7. The command that runs automatically whenever the container spins up
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]