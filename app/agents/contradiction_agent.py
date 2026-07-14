from typing import List
from pydantic import BaseModel, Field
from app.agents.state import ResearchState
from app.agents.llm import generate_structured_response

class ContradictionSchema(BaseModel):
    conflict_topic: str = Field(description="The topic or finding where sources disagree.")
    finding_a: str = Field(description="Finding from source group A.")
    source_ids_a: List[str] = Field(description="Source IDs for group A.")
    finding_b: str = Field(description="Opposing finding from source group B.")
    source_ids_b: List[str] = Field(description="Source IDs for group B.")
    explanation: str = Field(description="Brief explanation of the conflict, difference in methodology, or nuance.")

class ContradictionsListSchema(BaseModel):
    contradictions: List[ContradictionSchema] = Field(description="List of identified contradictions, or empty list if no direct conflicts exist.")

def contradiction_node(state: ResearchState) -> dict:
    """Agent 4 Node: Audits synthesis and retrieval context for contradictions or debates."""
    question = state["question"]
    context_str = state.get("context_str") or ""
    executive_summary = state.get("executive_summary") or ""
    claims = state.get("claims") or []
    
    if not context_str or "No relevant" in context_str or "Retrieval failed" in context_str:
        return {
            "contradictions": []
        }
        
    prompt = f"""
    User Query: {question}
    
    Retrieved Context:
    {context_str}
    
    Synthesized Findings:
    Executive Summary: {executive_summary}
    Claims: {[c.get('claim', '') for c in claims]}
    
    Examine the retrieved context and findings to identify any direct contradictions, conflicts, discrepancies,
    or differences in conclusions/methodologies between papers (e.g., Paper A finds X improves accuracy, while Paper B finds X decreases accuracy).
    """
    sys_instruction = (
        "You are the Contradiction Agent. Analyze scientific context and synthesized research to find disagreements, "
        "varying benchmarks, or conflicting reports. Only note actual conflicts, or return an empty contradictions list if none exist."
    )
    
    try:
        res = generate_structured_response(prompt, ContradictionsListSchema, system_instruction=sys_instruction)
        
        contradictions_dict = [
            {
                "conflict_topic": c.conflict_topic,
                "finding_a": c.finding_a,
                "source_ids_a": c.source_ids_a,
                "finding_b": c.finding_b,
                "source_ids_b": c.source_ids_b,
                "explanation": c.explanation
            }
            for c in res.contradictions
        ]
        
        return {
            "contradictions": contradictions_dict
        }
    except Exception as e:
        print(f"Contradiction Agent node error: {e}")
        return {
            "contradictions": []
        }
