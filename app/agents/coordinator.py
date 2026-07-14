from typing import Dict, Any
from sqlalchemy.orm import Session
from app.agents.graph import research_graph

def run_research_pipeline(db: Session, question: str) -> Dict[str, Any]:
    """
    Coordinates the entire multi-agent research pipeline using LangGraph.
    Invokes the compiled state graph sequentially.
    """
    # 1. Initialize Graph State
    initial_state = {
        "question": question
    }
    
    # 2. Execute Graph
    final_state = research_graph.invoke(initial_state)
    
    # 3. Retrieve final state outputs
    sources = final_state.get("sources") or []
    
    # Handle case where no sources were retrieved
    if not sources:
        return {
            "question": question,
            "answer": "No relevant documents or pages were found in the database. Please upload papers first.",
            "summary": "No documents found.",
            "claims": [],
            "contradictions": [],
            "citations": [],
            "report_markdown": "No documents found in database.",
            "sources": []
        }
        
    return {
        "question": question,
        "answer": final_state.get("formatted_synthesis") or "",
        "summary": final_state.get("executive_summary") or "",
        "claims": final_state.get("formatted_claims") or [],
        "contradictions": final_state.get("formatted_contradictions") or [],
        "citations": final_state.get("citations_list") or [],
        "report_markdown": final_state.get("report_markdown") or "",
        "sources": sources
    }
