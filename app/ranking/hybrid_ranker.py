import math
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.database import Paper, Page

def calculate_hybrid_scores(
    db: Session, 
    search_results: List[Dict[str, Any]], 
    current_year: int = 2026
) -> List[Dict[str, Any]]:
    """
    Ranks search results using the hybrid formula:
    final_score = 0.5 * bm25 + 0.2 * citation_score + 0.2 * recency_score + 0.1 * importance
    """
    if not search_results:
        return []

    # 1. Gather all Paper objects corresponding to retrieved pages to get metadata
    paper_ids = list(set(r["paper_id"] for r in search_results))
    papers_db = db.query(Paper).filter(Paper.id.in_(paper_ids)).all()
    papers_map = {p.id: p for p in papers_db}

    # Find max citations in retrieved set for normalization
    max_citations = max((p.citation_count for p in papers_db), default=0)
    max_citations = max(max_citations, 1) # Avoid division by zero

    # Find max BM25 score in retrieved set for normalization
    max_bm25 = max((r.get("score", 0.0) for r in search_results), default=0.0)
    max_bm25 = max(max_bm25, 0.0001)

    ranked_results = []

    for item in search_results:
        paper_id = item["paper_id"]
        page_no = item["page_no"]
        bm25_raw = item.get("score", 0.0)
        text = item["text"]

        paper = papers_map.get(paper_id)
        if not paper:
            continue

        # A. Normalized BM25 [0, 1]
        bm25_norm = bm25_raw / max_bm25

        # B. Citation Score (Log scaled to handle citation power-law)
        citation_count = paper.citation_count or 0
        citation_score = math.log1p(citation_count) / math.log1p(max_citations)

        # C. Recency Score (Linear decay over 50 years)
        pub_year = paper.year or 2020
        years_old = max(0, current_year - pub_year)
        recency_score = max(0.0, 1.0 - (years_old / 50.0))

        # D. Page Importance
        # Base importance decays by page number (intro/first page is higher)
        importance_base = 1.0 / (page_no ** 0.5)
        importance_base = max(0.2, min(1.0, importance_base))

        # Give a boost if section headers like 'conclusion' or 'results' are found on page
        header_boost = 0.0
        text_lower = text.lower()
        # Look for headers in the first 200 chars of page (where headers usually live)
        header_area = text_lower[:200]
        if any(h in header_area for h in ["conclusion", "concluding", "summary", "discussion", "abstract"]):
            header_boost = 0.2
            
        importance = min(1.0, importance_base + header_boost)

        # E. Combine Scores
        final_score = (
            0.5 * bm25_norm +
            0.2 * citation_score +
            0.2 * recency_score +
            0.1 * importance
        )

        ranked_results.append({
            "paper_id": paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "page_no": page_no,
            "text": text,
            "bm25_score": bm25_raw,
            "citation_score": citation_score,
            "recency_score": recency_score,
            "importance_score": importance,
            "final_score": final_score
        })

    # Sort results by final score descending
    ranked_results.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked_results
