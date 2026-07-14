from typing import TypedDict, List, Dict, Any, Optional

class ResearchState(TypedDict):
    """Represents the shared memory state of the PageRAG LangGraph research workflow."""
    question: str
    query_type: Optional[str]
    keywords: Optional[List[str]]
    search_queries: Optional[List[str]]
    context_str: Optional[str]
    sources: Optional[List[Dict[str, Any]]]
    executive_summary: Optional[str]
    claims: Optional[List[Dict[str, Any]]]
    synthesis_markdown: Optional[str] # Added for explicit state mapping
    contradictions: Optional[List[Dict[str, Any]]]
    formatted_synthesis: Optional[str]
    formatted_claims: Optional[List[Dict[str, Any]]]
    formatted_contradictions: Optional[List[Dict[str, Any]]]
    citations_list: Optional[List[Dict[str, Any]]]
    report_markdown: Optional[str]
