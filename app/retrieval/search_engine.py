import os
import re
import time
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session
from app.database import Page, Paper

# Elasticsearch Config
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX_NAME = "pagerag_pages"

class SearchEngine:
    def __init__(self):
        self.use_es = False
        self.es_client = None
        self.last_conn_attempt = 0
        self.conn_attempt_interval = 30  # Try connecting at most once every 30 seconds
        
        # Check if we should attempt Elasticsearch
        self._check_and_connect()

        # Local cache for BM25 fallback
        self.local_bm25 = None
        self.local_pages_cache = [] # List of dicts representing cached pages

    def _check_and_connect(self) -> bool:
        """Attempts to connect to Elasticsearch if not already connected."""
        if self.use_es and self.es_client:
            return True
            
        current_time = time.time()
        if current_time - self.last_conn_attempt < self.conn_attempt_interval:
            return False
            
        self.last_conn_attempt = current_time
        try:
            # Recreate client to ensure clean state
            self.es_client = Elasticsearch(ES_URL, request_timeout=5)
            if self.es_client.ping():
                self.use_es = True
                print(f"Successfully connected to Elasticsearch at {ES_URL}")
                self._init_es_index()
                return True
            else:
                print("Elasticsearch ping failed. Using local BM25 fallback.")
        except Exception as e:
            print(f"Could not connect to Elasticsearch: {e}. Using local BM25 fallback.")
        return False

    def _init_es_index(self):
        """Initializes the Elasticsearch index with appropriate mappings."""
        if not self.use_es or not self.es_client:
            return
        
        if not self.es_client.indices.exists(index=INDEX_NAME):
            mapping = {
                "mappings": {
                    "properties": {
                        "paper_id": {"type": "integer"},
                        "page_no": {"type": "integer"},
                        "text": {"type": "text", "analyzer": "english"}
                    }
                }
            }
            self.es_client.indices.create(index=INDEX_NAME, body=mapping)
            print(f"Created Elasticsearch index: {INDEX_NAME}")

    def index_page(self, db: Session, paper_id: int, page_no: int, text: str):
        """Indexes a page in Elasticsearch and invalidates local BM25 cache."""
        self._check_and_connect()
        # 1. Elasticsearch indexing
        if self.use_es and self.es_client:
            try:
                doc_id = f"{paper_id}_{page_no}"
                doc = {
                    "paper_id": paper_id,
                    "page_no": page_no,
                    "text": text
                }
                self.es_client.index(index=INDEX_NAME, id=doc_id, body=doc, refresh=True)
            except Exception as e:
                print(f"Elasticsearch indexing failed: {e}. Falling back to DB-only.")
        
        # Invalidate local cache
        self.local_bm25 = None
        self.local_pages_cache = []

    def delete_paper(self, paper_id: int):
        """Deletes a paper's pages from the index."""
        self._check_and_connect()
        if self.use_es and self.es_client:
            try:
                query = {
                    "query": {
                        "term": {
                            "paper_id": paper_id
                        }
                    }
                }
                self.es_client.delete_by_query(index=INDEX_NAME, body=query, refresh=True)
            except Exception as e:
                print(f"Elasticsearch delete failed: {e}")
        
        self.local_bm25 = None
        self.local_pages_cache = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for the local BM25 fallback."""
        return re.findall(r"\w+", text.lower())

    def _build_local_bm25(self, db: Session):
        """Loads pages from DB and constructs BM25 index."""
        db_pages = db.query(Page).all()
        if not db_pages:
            self.local_pages_cache = []
            self.local_bm25 = None
            return

        self.local_pages_cache = [
            {
                "paper_id": p.paper_id,
                "page_no": p.page_no,
                "text": p.text
            }
            for p in db_pages
        ]
        
        tokenized_corpus = [self._tokenize(p["text"]) for p in self.local_pages_cache]
        self.local_bm25 = BM25Okapi(tokenized_corpus)

    def search(self, db: Session, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Searches for relevant pages.
        Returns a list of dicts: [{"paper_id": 123, "page_no": 5, "text": "...", "score": 12.3}]
        """
        self._check_and_connect()
        if self.use_es and self.es_client:
            try:
                body = {
                    "size": top_k,
                    "query": {
                        "match": {
                            "text": query
                        }
                    }
                }
                res = self.es_client.search(index=INDEX_NAME, body=body)
                hits = res["hits"]["hits"]
                
                results = []
                for hit in hits:
                    source = hit["_source"]
                    results.append({
                        "paper_id": source["paper_id"],
                        "page_no": source["page_no"],
                        "text": source["text"],
                        "score": hit["_score"]
                    })
                return results
            except Exception as e:
                print(f"Elasticsearch search failed: {e}. Falling back to local BM25.")
        
        # Local BM25 Fallback
        if not self.local_bm25 or not self.local_pages_cache:
            self._build_local_bm25(db)
            
        if not self.local_bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.local_bm25.get_scores(tokenized_query)
        
        # Pair pages with scores and sort
        scored_pages = []
        for idx, score in enumerate(scores):
            if score > 0: # Only return pages with some match
                page_info = self.local_pages_cache[idx].copy()
                page_info["score"] = float(score)
                scored_pages.append(page_info)
                
        scored_pages.sort(key=lambda x: x["score"], reverse=True)
        return scored_pages[:top_k]

# Global search engine instance
search_engine = SearchEngine()
