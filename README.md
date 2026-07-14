# 📄 PageRAG: Citation-Aware Multi-Agent Research Assistant Engine

PageRAG is an enterprise-grade, citation-aware research engine that parses academic papers/PDFs, stores them in a hybrid search index, and answers complex queries using a multi-agent workflow orchestrated by **LangGraph** and **FastAPI**. 

It ensures **factual grounding** by cross-referencing claims directly against source documents, identifying contradictions, generating citation links, and computing evaluation metrics (Precision, Recall, Hallucination, Citation Accuracy) on the fly via an LLM Judge.

---

## 🚀 Key Features

*   **Multi-Agent Workflow (LangGraph)**: Sequential agentic pipeline for query parsing, retrieval, fact synthesis, contradiction resolving, citation verification, and report production.
*   **Hybrid Semantic & Lexical Search**: Multi-layered search fallback mechanism employing either Elasticsearch or a local fast BM25 implementation.
*   **Citation-Aware Synthesis**: Automatically maps assertions back to precise page numbers and source documents, rendering clean inline and bibliographic citations.
*   **Contradiction & Conflict Solver**: Reviews draft answers against source context to flag and resolve conflicting claims between different pages or papers.
*   **Built-in Evaluation Dashboard**: Computes real-time search engine precision/recall, citation accuracy, answer groundedness (hallucination score), and execution latency.
*   **Export Formats**: Generate publication-ready research reports in **PDF**, **Word (.docx)**, and **Markdown (.md)** formats.
*   **Database Agnostic**: Support for PostgreSQL and local SQLite storage with auto-migrated tables.
*   **Web Dashboard UI**: Beautiful, interactive web interface to upload PDFs, run queries, download reports, and visualize metrics.

---

## 🗺️ System Architecture

PageRAG coordinates several agents through a linear state graph powered by **LangGraph**.

```mermaid
graph TD
    START([User Query]) --> QU[Query Understanding Node]
    QU --> RET[Retrieval Node]
    RET --> RES[Research Node]
    RES --> CON[Contradiction Node]
    CON --> CIT[Citation Node]
    CIT --> REP[Report Node]
    REP --> END([Final Report & Metrics])

    subgraph State memory: ResearchState
        direction TB
        question[question]
        context_str[context_str]
        claims[claims]
        contradictions[contradictions]
        report_markdown[report_markdown]
    end
```

### Node Explanations
1.  **Query Understanding Agent**: Analyzes the query to identify search intent, extract keywords, and generate multiple optimized sub-queries.
2.  **Retrieval Agent**: Queries the index (Elasticsearch / BM25) and selects top-k relevant source pages.
3.  **Research Agent**: Performs deep facts extraction, synthesizes information, and compiles a narrative synthesis draft.
4.  **Contradiction Agent**: Audits findings to identify and flag direct contradictions or logical conflicts between source pages.
5.  **Citation Agent**: Cross-references claims with source text to verify grounding and map inline citations (`[1]`, `[2]`).
6.  **Report Agent**: Bundles the executive summary, narrative synthesis, verified citations, and contradictions into a formatted Markdown report.

---

## 🛠️ Step-by-Step Setup

### 1. Prerequisites
*   **Python 3.11+**
*   **Ollama** (running locally or in Docker)
*   **Docker & Docker Compose** (Optional, for full Elasticsearch & PostgreSQL stack)

### 2. Initialize Virtual Environment & Dependencies

```bash
# Clone the repository
git clone https://github.com/mahi3007/page_rag.git
cd page_rag

# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the example environment file:
```bash
cp .env.example .env
```

Open `.env` and specify your Ollama configuration:
```ini
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
```

---

## 🦙 Ollama Configuration

### Option A: Native installation (No Docker)
1. Download Ollama from [https://ollama.com](https://ollama.com).
2. Pull the default model:
   ```bash
   ollama pull llama3.2:latest
   ```

### Option B: Running Ollama in Docker (Recommended)
Uncomment GPU settings in `docker/docker-compose.yml` if you have NVIDIA GPU runtime configured:
```bash
# Start Ollama service container
docker-compose -f docker/docker-compose.yml up -d ollama

# Pull the model inside the container
docker exec -it pagerag_ollama ollama pull llama3.2:latest
```

---

## 🚀 Running the Server

Start the FastAPI application:
```bash
python main.py
```
Or run with Uvicorn reload enabled:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

*   **Web Dashboard UI**: [http://localhost:8000/](http://localhost:8000/)
*   **Interactive API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Testing

Execute the test suite with:
```bash
python -m pytest
```

---

## 🐳 Production Deployment (Full Stack)

To run the complete production stack (PostgreSQL, Elasticsearch, and Ollama) in Docker:

```bash
# Spin up the containers
docker-compose -f docker/docker-compose.yml up -d

# Pull model for Ollama
docker exec -it pagerag_ollama ollama pull llama3.2:latest
```

Modify your local `.env` settings to point to the docker services:
```ini
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/pagerag
ELASTICSEARCH_URL=http://localhost:9200
OLLAMA_URL=http://localhost:11434
```

---

## 🔌 API Endpoints Summary

### Document Ingestion & Search
*   `POST /api/upload`: Ingest research paper PDFs, extract metadata and pages, and index text in the search engine.
*   `POST /api/search`: Query the indexed text using hybrid scoring and retrieve ranked relevant sections.
*   `GET /api/papers`: List all uploaded papers.
*   `DELETE /api/papers/{paper_id}`: Remove a paper from both the database and the search index.

### Research Pipeline
*   `POST /api/ask`: Trigger the multi-agent LangGraph workflow. It runs query understanding, retrieval, synthesis, contradiction audit, and citation matching, returning the final report JSON.
*   `POST /api/report`: Exporters to compile query reports in `.pdf`, `.docx`, or `.md` formats.
*   `GET /api/metrics`: Retrieve aggregated evaluation analytics (precision, recall, latency, citation accuracy, and hallucination scores) alongside historic queries.

---

## 📁 Directory Structure

```
page_rag/
├── app/
│   ├── agents/          # LangGraph agents (query, retrieval, research, contradiction, etc.)
│   ├── api/             # FastAPI routing and route handlers
│   ├── evaluation/      # LLM-in-the-loop evaluation metrics
│   ├── ranking/         # Hybrid scoring modules (TF-IDF/BM25)
│   ├── reports/         # Document generators (PDF & Word exporters)
│   ├── retrieval/       # PDF parsing, OCR, and Search indexing engines
│   ├── templates/       # Dashboard HTML UI
│   └── database.py      # SQLAlchemy models & DB connectors
├── docker/              # Docker Compose and container configs
├── tests/               # Pytest test suite
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
└── .gitignore           # Ignored temporary/local files
```

---

## 📜 License
This project is open-source and available under the MIT License.
