from typing import List
from pydantic import BaseModel, Field
from app.agents.state import ResearchState
from app.agents.llm import generate_structured_response

class QueryAnalysisSchema(BaseModel):
    query_type: str = Field(description="Type of query: 'comparison', 'explanation', 'disagreement', 'synthesis', or 'other'")
    keywords: List[str] = Field(description="Search keywords extracted from the user query")
    search_queries: List[str] = Field(description="List of 2-3 expanded search query variations to search the page index")

def query_understanding_node(state: ResearchState) -> dict:
    """Agent 1 Node: Analyzes query and yields search parameters."""
    question = state["question"]
    prompt = f"Analyze the following user research query: '{question}'"
    sys_instruction = (
        "You are the Query Understanding Agent. Classify the user query and extract key search keywords "
        "and alternative query formulations to maximize information retrieval recall from scientific papers."
    )
    
    try:
        res = generate_structured_response(prompt, QueryAnalysisSchema, system_instruction=sys_instruction)
        return {
            "query_type": res.query_type,
            "keywords": res.keywords,
            "search_queries": res.search_queries
        }
    except Exception as e:
        print(f"Query Agent encountered error: {e}. Falling back to default.")
        return {
            "query_type": "synthesis",
            "keywords": [question],
            "search_queries": [question]
        }
