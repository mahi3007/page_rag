from app.agents.state import ResearchState
from app.database import SessionLocal
from app.retrieval.context_builder import build_context

def retrieval_node(state: ResearchState) -> dict:
    """Agent 2 Node: Retrieval Agent. Queries search engines for top pages using expanded query terms."""
    search_queries = state.get("search_queries") or [state["question"]]
    db = SessionLocal()
    
    merged_results = []
    seen_pages = set()
    
    try:
        # Query for each alternative query formulation to get maximum coverage
        for q in search_queries[:2]:
            context_str, sources = build_context(db, q, top_k_retrieve=30, top_k_select=5)
            for s in sources:
                page_key = (s["paper_id"], s["page_no"])
                if page_key not in seen_pages:
                    seen_pages.add(page_key)
                    merged_results.append(s)
                    
        # Sort merged results by final_score and limit to top 4 pages for fast CPU inference
        merged_results.sort(key=lambda x: x["final_score"], reverse=True)
        selected_sources = merged_results[:4]
        
        # Re-build formatted context string
        context_parts = []
        for idx, s in enumerate(selected_sources, 1):
            s["source_index"] = idx  # Re-index
            s["source_id"] = f"Doc{s['paper_id']}_P{s['page_no']}"
            context_block = (
                f"--- START SOURCE [{idx}] ---\n"
                f"Source ID: {s['source_id']}\n"
                f"Paper Title: {s['title']}\n"
                f"Authors: {s['authors'] or 'Unknown'}\n"
                f"Year: {s['year'] or 'Unknown'}\n"
                f"Page Number: {s['page_no']}\n"
                f"Content:\n{s['text']}\n"
                f"--- END SOURCE [{idx}] ---\n"
            )
            context_parts.append(context_block)
            
        context_str = "\n".join(context_parts)
        return {
            "context_str": context_str,
            "sources": selected_sources
        }
    except Exception as e:
        print(f"Retrieval Agent node error: {e}")
        return {
            "context_str": "Retrieval failed.",
            "sources": []
        }
    finally:
        db.close()
