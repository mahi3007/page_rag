import re
import fitz  # PyMuPDF
from typing import Dict, Any, List, Tuple
from pydantic import BaseModel, Field
from app.agents.llm import generate_structured_response

class PaperMetadataSchema(BaseModel):
    title: str = Field(description="The formal title of the academic paper.")
    authors: List[str] = Field(description="List of authors of the academic paper.")
    year: int = Field(description="The publication year of the paper. Use 2024 or current year if unclear, or 0 if completely unknown.")
    keywords: List[str] = Field(description="List of 3-5 keywords or key phrases of the paper.")
    doi: str = Field(description="Digital Object Identifier (DOI) if present, else empty string.")
    citation_count: int = Field(description="An estimated citation count if indicated, or just return a default value like 10.")

def extract_pdf_pages(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF file.
    Returns a list of dicts: [{"page_no": 1, "text": "..."}]
    """
    doc = fitz.open(file_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        # Basic cleanup: remove null characters, strip spaces
        text = text.replace("\x00", "").strip()
        pages.append({
            "page_no": page_num + 1,
            "text": text
        })
    doc.close()
    return pages

def extract_metadata_heuristics(first_page_text: str, pdf_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts paper metadata using regex rules and standard PDF headers as a fallback.
    """
    title = pdf_metadata.get("title", "").strip()
    if not title or len(title) < 5:
        # Try to take the first non-empty line of the page
        lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
        title = lines[0] if lines else "Untitled Research Paper"

    author_raw = pdf_metadata.get("author", "").strip()
    authors = [a.strip() for a in author_raw.split(",")] if author_raw else ["Unknown Author"]

    # Search for year
    year = 2024
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", first_page_text)
    if year_match:
        year = int(year_match.group(1))

    # Search for DOI
    doi = ""
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", first_page_text, re.IGNORECASE)
    if doi_match:
        doi = doi_match.group(0)

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "keywords": ["Research", "PDF"],
        "doi": doi,
        "citation_count": 0
    }

def extract_metadata_llm(first_page_text: str, pdf_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses the configured LLM to extract high-quality structured metadata from the first page of the paper.
    Falls back to heuristics if the LLM call fails.
    """
    # Truncate text to avoid overly large prompt context
    sample_text = first_page_text[:4000]
    
    prompt = f"""
    You are an expert academic librarian. Analyze the following text extracted from the first page of a research paper, along with some PDF file metadata, and extract the academic metadata.
    
    PDF FILE METADATA:
    {pdf_metadata}
    
    EXTRACTED TEXT FROM FIRST PAGE:
    {sample_text}
    """
    
    try:
        res = generate_structured_response(
            prompt=prompt,
            response_schema=PaperMetadataSchema,
            system_instruction="Extract structured academic metadata from the provided first page and metadata of the PDF. Be accurate."
        )
        return {
            "title": res.title,
            "authors": res.authors,
            "year": res.year,
            "keywords": res.keywords,
            "doi": res.doi,
            "citation_count": res.citation_count
        }
    except Exception as e:
        print(f"LLM Metadata extraction failed. Falling back to heuristics. Error: {e}")
        return extract_metadata_heuristics(first_page_text, pdf_metadata)

def process_pdf(file_path: str, use_llm_metadata: bool = True) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Processes a PDF file, extracting metadata and all page text.
    Returns: (metadata_dict, page_list)
    """
    doc = fitz.open(file_path)
    pdf_meta = doc.metadata or {}
    doc.close()
    
    pages = extract_pdf_pages(file_path)
    if not pages:
        raise ValueError("PDF file is empty or could not be read.")
    
    first_page_text = pages[0]["text"]
    
    if use_llm_metadata:
        metadata = extract_metadata_llm(first_page_text, pdf_meta)
    else:
        metadata = extract_metadata_heuristics(first_page_text, pdf_meta)
        
    return metadata, pages
