"""
Embedding and Vector Store for RAG
Uses simple TF-IDF for demo (can be upgraded to sentence transformers)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import math

class SimpleEmbeddings:
    """Simple TF-IDF based embeddings for RAG."""
    
    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.documents = []
        
    def fit(self, documents: List[str]):
        """Build vocabulary and compute IDF."""
        self.documents = documents
        
        # Build vocabulary
        all_words = set()
        for doc in documents:
            words = self._tokenize(doc)
            all_words.update(words)
        
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        
        # Compute IDF
        doc_freq = Counter()
        for doc in documents:
            words = set(self._tokenize(doc))
            doc_freq.update(words)
        
        n_docs = len(documents)
        self.idf = {
            word: math.log(n_docs / (freq + 1))
            for word, freq in doc_freq.items()
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def embed(self, text: str) -> np.ndarray:
        """Create TF-IDF embedding for text."""
        words = self._tokenize(text)
        word_counts = Counter(words)
        
        # TF-IDF vector
        vector = np.zeros(len(self.vocabulary))
        for word, count in word_counts.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tf = count / len(words) if words else 0
                idf = self.idf.get(word, 0)
                vector[idx] = tf * idf
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return np.dot(vec1, vec2)

class VectorStore:
    """Simple vector store for retrieval."""
    
    def __init__(self):
        self.embeddings_model = SimpleEmbeddings()
        self.vectors = []
        self.metadata = []
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to vector store."""
        # Extract text for embedding
        texts = [self._doc_to_text(doc) for doc in documents]
        
        # Fit embeddings
        self.embeddings_model.fit(texts)
        
        # Create vectors
        self.vectors = [self.embeddings_model.embed(text) for text in texts]
        self.metadata = documents
    
    def _doc_to_text(self, doc: Dict) -> str:
        """Convert document to text for embedding."""
        parts = [
            doc.get("name", ""),
            doc.get("description", ""),
            doc.get("type", ""),
            doc.get("category", ""),
            " ".join(doc.get("skills_measured", []))
        ]
        return " ".join(parts)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar documents."""
        query_vec = self.embeddings_model.embed(query)
        
        # Compute similarities
        similarities = [
            (doc, self.embeddings_model.similarity(query_vec, vec))
            for doc, vec in zip(self.metadata, self.vectors)
        ]
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

if __name__ == "__main__":
    # Test
    catalog_path = Path(__file__).parent.parent / "data" / "processed" / "assessment_catalog.json"
    with open(catalog_path) as f:
        catalog = json.load(f)
    
    store = VectorStore()
    store.add_documents(catalog)
    
    results = store.search("analytical problem solving", top_k=3)
    print("Top 3 results:")
    for doc, score in results:
        print(f"  {doc['name']}: {score:.3f}")
