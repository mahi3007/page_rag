import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Determine database URL. Fallback to SQLite if not provided or PostgreSQL connection fails.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///pagerag.db")

# SQLite connection args (required for sqlite multithreading)
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# Initialize engine and session
try:
    engine = create_engine(DATABASE_URL, connect_args=connect_args)
    # Test connection
    with engine.connect() as conn:
        pass
except Exception as e:
    print(f"Warning: Failed to connect to database at {DATABASE_URL}. Error: {e}")
    print("Falling back to local SQLite database: sqlite:///pagerag.db")
    DATABASE_URL = "sqlite:///pagerag.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    authors = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    keywords = Column(String, nullable=True)
    doi = Column(String, nullable=True)
    citation_count = Column(Integer, default=0)

    # Relationships
    pages = relationship("Page", back_populates="paper", cascade="all, delete-orphan")

class Page(Base):
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    page_no = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

    # Relationships
    paper = relationship("Paper", back_populates="pages")

class QueryRecord(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, nullable=False)
    answer = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    evaluation = relationship("Evaluation", back_populates="query", uselist=False, cascade="all, delete-orphan")

class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("queries.id", ondelete="CASCADE"), nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)
    citation_accuracy = Column(Float, nullable=False)
    hallucination_score = Column(Float, nullable=False)

    # Relationships
    query = relationship("QueryRecord", back_populates="evaluation")

# Helper to initialize the database
def init_db():
    Base.metadata.create_all(bind=engine)

# Dependency to get db session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
