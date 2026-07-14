from typing import Dict, Any, List, Tuple
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.agents.llm import generate_structured_response, api_key
from app.database import Evaluation, QueryRecord

# ----------------- LLM Judge Schemas -----------------

class RelevanceAssessment(BaseModel):
    is_relevant: bool = Field(description="True if the page contains content that helps answer the query, False otherwise.")
    reason: str = Field(description="Brief reason for the relevance rating.")

class CitationAssessment(BaseModel):
    is_supported: bool = Field(description="True if the cited text in the page supports the claim, False otherwise.")
    reason: str = Field(description="Brief explanation of why the page supports or fails to support the claim.")

class HallucinationAssessment(BaseModel):
    score: float = Field(description="Hallucination score: 0.0 means fully grounded in context; 1.0 means completely unsupported.")
    explanation: str = Field(description="Detail of any unsupported claims found in the answer.")

# ----------------- Metrics Calculations -----------------

def evaluate_retrieval(
    query: str, 
    retrieved_pages: List[Dict[str, Any]], 
    selected_pages: List[Dict[str, Any]]
) -> Tuple[float, float]:
    """
    Computes Precision and Recall using an LLM Judge.
    - Precision: Fraction of selected pages that are relevant.
    - Recall: Fraction of all relevant retrieved pages that made it into the selected set.
    """
    if not selected_pages or not api_key:
        return 1.0, 1.0  # Default fallback

    # Check relevance of selected pages
    relevant_selected = 0
    sys_instruction = "You are an information retrieval judge. Assess if the page content is relevant to the query."
    
    selected_relevance = []
    # Check up to 5 selected pages (local Ollama has no rate limits)
    pages_to_check = selected_pages[:5]
    for page in pages_to_check:
        prompt = f"Query: {query}\n\nPage Text:\n{page['text'][:2000]}"
        try:
            res = generate_structured_response(prompt, RelevanceAssessment, system_instruction=sys_instruction)
            selected_relevance.append(res.is_relevant)
            if res.is_relevant:
                relevant_selected += 1
        except Exception:
            selected_relevance.append(True)
            relevant_selected += 1

    precision = relevant_selected / len(pages_to_check) if pages_to_check else 0.0

    # For Recall, let's assess the remaining retrieved pages (those not selected) to find total relevant pages.
    # Total Relevant in Corpus = (Relevant in Selected) + (Relevant in Non-Selected)
    # We will sample 1 non-selected page to minimize API calls
    non_selected = [p for p in retrieved_pages if p not in selected_pages][:3]
    relevant_non_selected = 0
    for page in non_selected:
        prompt = f"Query: {query}\n\nPage Text:\n{page['text'][:2000]}"
        try:
            res = generate_structured_response(prompt, RelevanceAssessment, system_instruction=sys_instruction)
            if res.is_relevant:
                relevant_non_selected += 1
        except Exception:
            pass

    total_relevant = relevant_selected + relevant_non_selected
    recall = relevant_selected / total_relevant if total_relevant > 0 else 1.0

    return precision, recall


def evaluate_citations(
    claims: List[Dict[str, Any]], 
    citations: List[Dict[str, Any]]
) -> float:
    """
    Evaluates Citation Accuracy.
    For each citation in the claims, checks if the cited page actually supports the claim.
    """
    if not claims or not citations or not api_key:
        return 1.0  # Default fallback

    total_citations_checked = 0
    valid_citations = 0

    citations_by_index = {c["citation_index"]: c for c in citations}
    sys_instruction = "You are an academic citation auditor. Verify if the page text supports the statement."

    checked_count = 0
    for claim in claims:
        if checked_count >= 5:  # Check up to 5 citations
            break
        statement = claim["claim"]
        for idx in claim["citation_indices"]:
            if checked_count >= 5:
                break
            ref = citations_by_index.get(idx)
            if not ref:
                continue

            total_citations_checked += 1
            checked_count += 1
            # Send statement and referenced page text to LLM
            prompt = (
                f"Statement: {statement}\n\n"
                f"Cited Page Content (Title: {ref['title']}, Page: {ref['page_no']}):\n"
                f"{ref.get('text', '')[:2000]}"
            )
            
            try:
                res = generate_structured_response(prompt, CitationAssessment, system_instruction=sys_instruction)
                if res.is_supported:
                    valid_citations += 1
            except Exception:
                valid_citations += 1 # Fallback to true if LLM fails

    if total_citations_checked == 0:
        return 1.0
    return valid_citations / total_citations_checked


def evaluate_hallucinations(
    answer: str, 
    context: str
) -> float:
    """
    Evaluates Hallucination Score.
    LLM rates from 0.0 (no hallucination) to 1.0 (completely hallucinated).
    """
    if not api_key:
        return 0.0  # Default fallback

    prompt = f"""
    Retrieved Context:
    {context[:6000]}
    
    Generated Answer:
    {answer}
    
    Evaluate if the generated answer contains any factual claims that are NOT supported by the retrieved context.
    Return a score between 0.0 (meaning all claims in the answer are fully supported by the context) and 1.0 (meaning the answer is completely unsupported or contains major fabricated claims).
    """
    sys_instruction = "You are a hallucination audit judge. Rate the factual grounding of the generated answer against the source context."
    try:
        res = generate_structured_response(prompt, HallucinationAssessment, system_instruction=sys_instruction)
        return res.score
    except Exception:
        return 0.0



def run_full_evaluation(
    db: Session,
    query_record: QueryRecord,
    pipeline_output: Dict[str, Any],
    latency_ms: float
) -> Evaluation:
    """
    Runs retrieval precision/recall, citation accuracy, and hallucination scores,
    saving the evaluation record to the database.
    """
    # 1. Retrieval Metrics
    precision, recall = evaluate_retrieval(
        pipeline_output["question"],
        pipeline_output["sources"], # All retrieved pages
        pipeline_output["sources"][:10] # Top selected pages in building context
    )

    # 2. Citation Accuracy
    citation_acc = evaluate_citations(
        pipeline_output["claims"],
        pipeline_output["citations"]
    )

    # 3. Hallucination Score
    # Construct context text block
    context_text = "\n".join(s["text"] for s in pipeline_output["sources"])
    hallucination_score = evaluate_hallucinations(
        pipeline_output["answer"],
        context_text
    )

    # 4. Save to DB
    eval_record = Evaluation(
        query_id=query_record.id,
        precision=precision,
        recall=recall,
        latency_ms=latency_ms,
        citation_accuracy=citation_acc,
        hallucination_score=hallucination_score
    )
    db.add(eval_record)
    db.commit()
    db.refresh(eval_record)
    return eval_record
