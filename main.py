import os
import warnings

# Set environment variable early so uvicorn's reload worker sub-processes inherit it
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress all warnings globally in this process
warnings.simplefilter("ignore")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.database import init_db
from app.api.endpoints import router as api_router


# Use lifespan context manager instead of deprecated @app.on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    port = int(os.getenv("PORT", 8000))
    print("Database tables initialized.")
    print(f"\n* PageRAG is running on http://localhost:{port} *\n")
    yield
    # Shutdown (add any cleanup here if needed)


app = FastAPI(
    title="PageRAG API",
    description="Citation-Aware Multi-Agent Research Assistant Engine",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API router
app.include_router(api_router, prefix="/api")

# Serve frontend HTML directly at root
@app.get("/", response_class=HTMLResponse)
async def read_index():
    template_path = os.path.join(os.path.dirname(__file__), "app", "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return """
        <html>
            <head><title>PageRAG Assistant</title></head>
            <body style='font-family: Arial, sans-serif; text-align: center; padding-top: 100px;'>
                <h1>Welcome to PageRAG</h1>
                <p>Frontend template not found. Please verify app/templates/index.html exists.</p>
                <p>Go to <a href='/docs'>/docs</a> to view API documentation.</p>
            </body>
        </html>
        """

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="warning"
    )
