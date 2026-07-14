from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from app.retrieval.search_engine import search_engine
from app.ranking.hybrid_ranker import calculate_hybrid_scores

def build_context(
    db: Session, 
    query: str, 
    top_k_retrieve: int = 40, 
    top_k_select: int = 15
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieves pages, ranks them, selects the top N, and formats them into a structured context.
    Returns:
        context_str: A string of formatted page content for the LLM
        sources: A list of dicts containing the selected sources
    """
    # 1. Search Elasticsearch or local index
    raw_results = search_engine.search(db, query, top_k=top_k_retrieve)
    if not raw_results:
        return "No relevant documents found.", []

    # 2. Rank using Hybrid Ranking Engine
    ranked_results = calculate_hybrid_scores(db, raw_results)

    # 3. Select top N results
    selected_results = ranked_results[:top_k_select]

    # 4. Construct context string
    context_parts = []
    sources = []
    
    for idx, res in enumerate(selected_results, 1):
        source_id = f"Doc{res['paper_id']}_P{res['page_no']}"
        
        # Format the context block for this page
        context_block = (
            f"--- START SOURCE [{idx}] ---\n"
            f"Source ID: {source_id}\n"
            f"Paper Title: {res['title']}\n"
            f"Authors: {res['authors'] or 'Unknown'}\n"
            f"Year: {res['year'] or 'Unknown'}\n"
            f"Page Number: {res['page_no']}\n"
            f"Content:\n{res['text']}\n"
            f"--- END SOURCE [{idx}] ---\n"
        )
        context_parts.append(context_block)
        
        # Keep metadata about the source for citation generation
        sources.append({
            "source_index": idx,
            "source_id": source_id,
            "paper_id": res["paper_id"],
            "title": res["title"],
            "authors": res["authors"],
            "year": res["year"],
            "page_no": res["page_no"],
            "text": res["text"],
            "final_score": res["final_score"]
        })

    context_str = "\n".join(context_parts)
    return context_str, sources
