import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base, Paper, Page
from app.ranking.hybrid_ranker import calculate_hybrid_scores
from app.retrieval.search_engine import SearchEngine

# Setup mock in-memory database for testing
@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_db_models(db_session):
    """Test that SQLAlchemy models can be created and queried successfully."""
    # Create a paper
    paper = Paper(
        title="Test Attention",
        authors="Test Vaswani",
        year=2017,
        keywords="transformer, attention",
        doi="10.1234/test",
        citation_count=50
    )
    db_session.add(paper)
    db_session.commit()
    db_session.refresh(paper)

    assert paper.id is not None
    assert paper.title == "Test Attention"

    # Create a page
    page = Page(
        paper_id=paper.id,
        page_no=1,
        text="This is a test text containing self-attention."
    )
    db_session.add(page)
    db_session.commit()

    assert page.id is not None
    assert len(paper.pages) == 1
    assert paper.pages[0].text == "This is a test text containing self-attention."

def test_tokenizer():
    """Test the SearchEngine's local BM25 fallback tokenizer."""
    engine = SearchEngine()
    tokens = engine._tokenize("Attention IS All You Need! 123.")
    assert tokens == ["attention", "is", "all", "you", "need", "123"]

def test_hybrid_ranking(db_session):
    """Test that the hybrid ranker applies correct score formulas."""
    # 1. Add mock papers
    paper1 = Paper(id=1, title="Paper A", authors="Author A", year=2024, citation_count=100) # Recent, highly cited
    paper2 = Paper(id=2, title="Paper B", authors="Author B", year=2000, citation_count=0)   # Old, uncited
    db_session.add_all([paper1, paper2])
    db_session.commit()

    # 2. Mock search results (Page 1 of paper 1, Page 20 of paper 2)
    search_results = [
        {
            "paper_id": 1,
            "page_no": 1,
            "text": "This is page 1 of paper A. Abstract and introduction here.",
            "score": 10.0 # raw BM25 score
        },
        {
            "paper_id": 2,
            "page_no": 20,
            "text": "This is page 20 of paper B. Basic body text details.",
            "score": 5.0 # raw BM25 score
        }
    ]

    # 3. Calculate hybrid scores (setting current year to 2026 for consistency)
    ranked = calculate_hybrid_scores(db_session, search_results, current_year=2026)

    assert len(ranked) == 2
    
    # Paper A should rank first because of higher BM25, higher recency (2024 vs 2000),
    # higher citations (100 vs 0), and higher page importance (page 1 vs page 20)
    assert ranked[0]["paper_id"] == 1
    assert ranked[1]["paper_id"] == 2
    
    # Verify scores are computed and exist
    assert "final_score" in ranked[0]
    assert ranked[0]["final_score"] > ranked[1]["final_score"]
    
    # Check that linear decay for recency is clamped and correct
    # Paper 2 is from year 2000. 2026 - 2000 = 26 years old. Linear decay: 1.0 - (26/50) = 0.48.
    assert abs(ranked[1]["recency_score"] - 0.48) < 0.01
