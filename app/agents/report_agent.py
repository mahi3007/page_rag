from datetime import datetime
from app.agents.state import ResearchState

def report_node(state: ResearchState) -> dict:
    """Agent 6 Node: Report Compiler. Formats research results into a Markdown report."""
    query = state["question"]
    exec_summary = state.get("executive_summary") or "No summary available."
    synthesis_markdown = state.get("formatted_synthesis") or "No synthesis details."
    claims = state.get("formatted_claims") or []
    contradictions = state.get("formatted_contradictions") or []
    citations = state.get("citations_list") or []
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # 1. Title and Metadata
    report = f"# Research Report: {query}\n\n"
    report += f"**Compiled by PageRAG Research Agentic System**\n"
    report += f"**Date:** {current_date}\n\n"
    report += "---\n\n"
    
    # 2. Executive Summary
    report += "## 1. Executive Summary\n"
    report += f"{exec_summary}\n\n"
    
    # 3. Key Findings
    report += "## 2. Key Findings & Supporting Evidence\n"
    if not claims:
        report += "No major findings identified.\n\n"
    else:
        for idx, c in enumerate(claims, 1):
            citations_str = ", ".join(f"[{i}]" for i in c.get("citation_indices", []))
            report += f"### Finding {idx}: {c.get('claim')} {citations_str}\n"
            report += f"* **Evidence:** \"{c.get('evidence')}\"\n\n"
        
    # 4. Detailed Synthesis
    report += "## 3. Comprehensive Synthesis\n"
    report += f"{synthesis_markdown}\n\n"
    
    # 5. Contradictions & Nuance
    report += "## 4. Disagreements and Nuances\n"
    if not contradictions:
        report += "No major direct contradictions were identified in the corpus regarding this query.\n\n"
    else:
        for idx, con in enumerate(contradictions, 1):
            ref_a = ", ".join(f"[{i}]" for i in con.get("citation_indices_a", []))
            ref_b = ", ".join(f"[{i}]" for i in con.get("citation_indices_b", []))
            report += f"### Disagreement {idx}: {con.get('topic')}\n"
            report += f"* **Finding A:** {con.get('finding_a')} {ref_a}\n"
            report += f"* **Finding B:** {con.get('finding_b')} {ref_b}\n"
            report += f"* **Resolution/Nuance:** {con.get('explanation')}\n\n"
            
    # 6. References
    report += "## 5. References\n"
    if not citations:
        report += "No reference citations compiled.\n\n"
    else:
        for ref in citations:
            authors = ref.get("authors") or "Unknown"
            report += f"[{ref.get('citation_index')}] *{ref.get('title')}*, {authors} ({ref.get('year')}), Page {ref.get('page_no')}\n"
        
    return {
        "report_markdown": report
    }
