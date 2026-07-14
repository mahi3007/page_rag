from langgraph.graph import StateGraph, START, END
from app.agents.state import ResearchState
from app.agents.query_agent import query_understanding_node
from app.agents.retrieval_agent import retrieval_node
from app.agents.research_agent import research_node
from app.agents.contradiction_agent import contradiction_node
from app.agents.citation_agent import citation_node
from app.agents.report_agent import report_node

# Create the state graph
builder = StateGraph(ResearchState)

# Add all agent nodes to the graph
builder.add_node("query_understanding", query_understanding_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("research", research_node)
builder.add_node("contradiction", contradiction_node)
builder.add_node("citation", citation_node)
builder.add_node("report", report_node)

# Connect the nodes sequentially
builder.add_edge(START, "query_understanding")
builder.add_edge("query_understanding", "retrieval")
builder.add_edge("retrieval", "research")
builder.add_edge("research", "contradiction")
builder.add_edge("contradiction", "citation")
builder.add_edge("citation", "report")
builder.add_edge("report", END)

# Compile the workflow graph
research_graph = builder.compile()
