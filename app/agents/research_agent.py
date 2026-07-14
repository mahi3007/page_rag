from typing import List
from pydantic import BaseModel, Field
from app.agents.state import ResearchState
from app.agents.llm import generate_response, generate_structured_response

class ResearchClaimSchema(BaseModel):
    claim: str = Field(description="A specific claim or key finding extracted from the text.")
    evidence: str = Field(description="Direct citation text or supporting detail from the pages.")
    source_ids: List[str] = Field(description="List of Source IDs supporting this claim (e.g. ['Doc1_P5']). Must match Source IDs in the context.")

class ClaimsListSchema(BaseModel):
    claims: List[ResearchClaimSchema] = Field(description="List of key claims extracted from the context.")

def research_node(state: ResearchState) -> dict:
    """Agent 3 Node: Performs deep research synthesis on retrieved page content."""
    question = state["question"]
    context_str = state.get("context_str") or ""
    
    if not context_str or "No relevant" in context_str or "Retrieval failed" in context_str:
        return {
            "executive_summary": "No documents available for research.",
            "claims": [],
            "synthesis_markdown": "No pages retrieved."
        }
        
    # Step 1: Generate Synthesis narrative in plain text (extremely fast!)
    prompt_synthesis = f"""
    User Query: {question}
    
    You have access to the following retrieved pages from research papers:
    {context_str}
    
    Write a detailed synthesis narrative of the findings in Markdown. Cite source pages inline using brackets with the Source ID (e.g. "Some text [Doc3_P12]").
    """
    sys_instruction_synthesis = (
        "You are the Research Synthesis Agent. Your job is to read retrieved academic paper pages, synthesize the findings "
        "impartially, and write a high-quality Markdown response. You must cite exact pages "
        "using [DocX_PY] format."
    )
    synthesis_markdown = generate_response(prompt_synthesis, system_instruction=sys_instruction_synthesis)
    
    # Step 2: Generate a brief executive summary
    prompt_summary = f"""
    Summarize the following research synthesis into a single concise paragraph (executive summary):
    {synthesis_markdown}
    """
    sys_instruction_summary = "You are a research assistant. Write a short, single-paragraph executive summary of the provided text."
    executive_summary = generate_response(prompt_summary, system_instruction=sys_instruction_summary)

    # Step 3: Extract structured claims from the generated synthesis
    prompt_claims = f"""
    Context:
    {context_str}
    
    Synthesis:
    {synthesis_markdown}
    
    Based on the context and synthesis, extract 2-3 key claims/findings.
    For each claim, provide the claim text, direct evidence text from the context, and the list of Source IDs supporting it (e.g. ['Doc1_P5']).
    """
    sys_instruction_claims = "You are the Claims Extraction Agent. Extract a structured list of key findings and supporting evidence."
    
    try:
        res = generate_structured_response(prompt_claims, ClaimsListSchema, system_instruction=sys_instruction_claims)
        claims_dict = [
            {
                "claim": c.claim,
                "evidence": c.evidence,
                "source_ids": c.source_ids
            }
            for c in res.claims
        ]
    except Exception as e:
        print(f"Claims extraction failed: {e}. Falling back to empty list.")
        claims_dict = []
        
    return {
        "executive_summary": executive_summary,
        "claims": claims_dict,
        "synthesis_markdown": synthesis_markdown
    }
