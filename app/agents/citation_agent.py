import re
from app.agents.state import ResearchState

def citation_node(state: ResearchState) -> dict:
    """Agent 5 Node: Replaces temporary source tags with clean, sequential numeric brackets [1]."""
    synthesis_text = state.get("synthesis_markdown") or ""
    claims = state.get("claims") or []
    contradictions = state.get("contradictions") or []
    sources = state.get("sources") or []
    
    # 1. Parse all source_ids in the synthesis
    source_tag_pattern = r"\[Doc(\d+)_P(\d+)\]"
    found_tags = re.findall(source_tag_pattern, synthesis_text)
    
    source_key_to_idx = {}
    sources_by_id = {f"Doc{s['paper_id']}_P{s['page_no']}": s for s in sources}
    
    mentioned_ids = set()
    for tag in found_tags:
        mentioned_ids.add(f"Doc{tag[0]}_P{tag[1]}")
    for claim in claims:
        for sid in claim.get("source_ids", []):
            mentioned_ids.add(sid)
    for contra in contradictions:
        for sid in contra.get("source_ids_a", []) + contra.get("source_ids_b", []):
            mentioned_ids.add(sid)
            
    idx_counter = 1
    citations_list = []
    
    # Map mentioned ones first
    for sid in list(mentioned_ids):
        if sid in sources_by_id:
            src = sources_by_id[sid]
            source_key = (src["paper_id"], src["page_no"])
            if source_key not in source_key_to_idx:
                source_key_to_idx[source_key] = idx_counter
                citations_list.append({
                    "citation_index": idx_counter,
                    "paper_id": src["paper_id"],
                    "title": src["title"],
                    "authors": src["authors"],
                    "year": src["year"],
                    "page_no": src["page_no"],
                    "source_id": sid
                })
                idx_counter += 1
                
    # Fallback to make sure every source has a key if not mentioned but retrieved
    for src in sources:
        source_key = (src["paper_id"], src["page_no"])
        if source_key not in source_key_to_idx:
            source_key_to_idx[source_key] = idx_counter
            citations_list.append({
                "citation_index": idx_counter,
                "paper_id": src["paper_id"],
                "title": src["title"],
                "authors": src["authors"],
                "year": src["year"],
                "page_no": src["page_no"],
                "source_id": src["source_id"]
            })
            idx_counter += 1

    # 2. Replace tags in synthesis text
    def replace_tag(match):
        p_id, p_no = match.group(1), match.group(2)
        key = (int(p_id), int(p_no))
        c_idx = source_key_to_idx.get(key)
        return f"[{c_idx}]" if c_idx else match.group(0)

    formatted_synthesis = re.sub(source_tag_pattern, replace_tag, synthesis_text)
    
    # 3. Format claims & contradictions with bibliography indices
    formatted_claims = []
    for claim in claims:
        mapped_indices = []
        for sid in claim.get("source_ids", []):
            match = re.match(r"Doc(\d+)_P(\d+)", sid)
            if match:
                key = (int(match.group(1)), int(match.group(2)))
                c_idx = source_key_to_idx.get(key)
                if c_idx:
                    mapped_indices.append(c_idx)
        formatted_claims.append({
            "claim": claim.get("claim"),
            "evidence": claim.get("evidence"),
            "citation_indices": sorted(list(set(mapped_indices)))
        })
        
    formatted_contradictions = []
    for contra in contradictions:
        indices_a = []
        indices_b = []
        for sid in contra.get("source_ids_a", []):
            match = re.match(r"Doc(\d+)_P(\d+)", sid)
            if match:
                key = (int(match.group(1)), int(match.group(2)))
                c_idx = source_key_to_idx.get(key)
                if c_idx:
                    indices_a.append(c_idx)
        for sid in contra.get("source_ids_b", []):
            match = re.match(r"Doc(\d+)_P(\d+)", sid)
            if match:
                key = (int(match.group(1)), int(match.group(2)))
                c_idx = source_key_to_idx.get(key)
                if c_idx:
                    indices_b.append(c_idx)
        
        formatted_contradictions.append({
            "topic": contra.get("topic"),
            "finding_a": contra.get("finding_a"),
            "citation_indices_a": sorted(list(set(indices_a))),
            "finding_b": contra.get("finding_b"),
            "citation_indices_b": sorted(list(set(indices_b))),
            "explanation": contra.get("explanation")
        })
        
    # Sort bibliography by citation index
    citations_list.sort(key=lambda x: x["citation_index"])
    
    return {
        "formatted_synthesis": formatted_synthesis,
        "formatted_claims": formatted_claims,
        "formatted_contradictions": formatted_contradictions,
        "citations_list": citations_list
    }
