# Running PageRAG

PageRAG is a Citation-Aware Multi-Agent Research Assistant Engine built with FastAPI, LangChain, and SQLAlchemy. It uses **Ollama** for all LLM inference, supports local BM25 search fallback, and automatic database fallback (SQLite/PostgreSQL).

---

## 📋 Prerequisites

Before running the project, ensure you have the following installed:
1. **Python 3.11+**
2. **Ollama** — either running natively or via the included Docker Compose file
3. **Docker & Docker Compose** (optional, for Ollama, Postgres, and Elasticsearch containers)

---

## 🛠️ Step-by-Step Setup

### 1. Initialize Virtual Environment

To isolate the dependencies, create and activate a Python virtual environment:

#### On Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### On macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Open `.env` and configure your Ollama settings:

```ini
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
```

Other popular models you can use: `mistral:latest`, `phi3:mini`, `llama3.1:8b`, `gemma2:9b`

---

## 🦙 Setting Up Ollama

### Option A — Native Ollama (no Docker)

1. Install Ollama from [https://ollama.com](https://ollama.com)
2. Pull and run your chosen model:
   ```bash
   ollama pull llama3.2:latest
   ```
3. Ollama starts automatically at `http://localhost:11434`.

### Option B — Ollama via Docker Compose (recommended)

The `docker/docker-compose.yml` includes a pre-configured Ollama service:

```bash
docker-compose -f docker/docker-compose.yml up -d ollama
```

Then pull a model inside the container:

```bash
docker exec -it pagerag_ollama ollama pull llama3.2:latest
```

> **GPU acceleration:** Uncomment the `deploy.resources` section in `docker/docker-compose.yml` to enable NVIDIA GPU support (requires `nvidia-container-toolkit`).

---

## 🚀 Running the Server

```bash
python main.py
```

Or run Uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Accessing the App
* **Web Frontend UI:** [http://localhost:8000/](http://localhost:8000/)
* **Interactive API Documentation (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Running the Test Suite

```bash
python -m pytest
```

---

## 🐳 Running with External Services (Optional)

By default, the app uses **SQLite** (`pagerag.db`) and **BM25** search. To spin up all external services (Ollama, PostgreSQL, Elasticsearch) in containers:

```bash
docker-compose -f docker/docker-compose.yml up -d
```

Then pull a model into the Ollama container:

```bash
docker exec -it pagerag_ollama ollama pull llama3.2:latest
```

### Configure `.env` for External Services

```ini
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/pagerag
ELASTICSEARCH_URL=http://localhost:9200
OLLAMA_URL=http://localhost:11434
```
