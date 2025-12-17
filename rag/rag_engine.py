"""
RAG Engine for Assessment Recommendations
Combines retrieval with generation for intelligent recommendations
"""

import json
from pathlib import Path
from typing import Dict, List
from .embeddings import VectorStore

class RAGEngine:
    """RAG-based recommendation engine."""
    
    def __init__(self, catalog_path: str):
        self.vector_store = VectorStore()
        self.load_catalog(catalog_path)
    
    def load_catalog(self, catalog_path: str):
        """Load and index assessment catalog."""
        with open(catalog_path) as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict) and "assessments" in data:
            catalog = data["assessments"]
        elif isinstance(data, list):
            catalog = data
        else:
            catalog = [data]
        
        self.vector_store.add_documents(catalog)
        print(f"✅ Loaded {len(catalog)} assessments into RAG engine")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant assessments."""
        results = self.vector_store.search(query, top_k=top_k)
        formatted_results = []
        for doc, score in results:
            if isinstance(doc, dict):
                formatted_results.append({"assessment": doc, "relevance_score": score})
            else:
                # Handle case where doc might be a string or other type
                formatted_results.append({"assessment": {"name": str(doc)}, "relevance_score": score})
        return formatted_results
    
    def generate_recommendation(self, query: str, preferences: Dict = None) -> Dict:
        """
        Generate recommendation using RAG approach.
        
        Args:
            query: Job description or requirements
            preferences: User preferences (fairness_weight, time_weight, etc.)
        
        Returns:
            Recommendation with explanation
        """
        preferences = preferences or {}
        
        # Retrieve relevant assessments
        retrieved = self.retrieve(query, top_k=10)
        
        # Extract assessments
        assessments = [r["assessment"] for r in retrieved]
        
        # Score and rank based on preferences
        scored = self._score_assessments(assessments, query, preferences)
        
        # Generate explanation
        top = scored[0] if scored else None
        explanation = self._generate_explanation(top, query, preferences) if top else {}
        
        return {
            "query": query,
            "top_recommendation": top,
            "alternatives": scored[1:4],
            "explanation": explanation,
            "retrieval_results": retrieved[:3]
        }
    
    def _score_assessments(self, assessments: List[Dict], query: str, preferences: Dict) -> List[Dict]:
        """Score assessments based on preferences."""
        fairness_weight = preferences.get("fairness_weight", 0.5)
        time_weight = preferences.get("time_weight", 0.3)
        
        scored = []
        for assessment in assessments:
            # Base score from validity
            validity = assessment.get("validity", 0.5)
            
            # Fairness score
            fairness_metrics = assessment.get("fairness_metrics", {})
            fairness_score = 1.0 - fairness_metrics.get("adverse_impact_risk", 0.5)
            
            # Time score (prefer shorter assessments)
            duration = assessment.get("duration_minutes", 30)
            time_score = max(0, 1.0 - (duration / 120))
            
            # Combined score
            total_score = (
                validity * 1.0 +
                fairness_score * fairness_weight +
                time_score * time_weight
            )
            
            assessment["recommendation_score"] = total_score
            scored.append(assessment)
        
        scored.sort(key=lambda x: x["recommendation_score"], reverse=True)
        return scored
    
    def _generate_explanation(self, assessment: Dict, query: str, preferences: Dict) -> Dict:
        """Generate explanation for recommendation."""
        return {
            "primary_reason": f"Best match for your requirements with score {assessment['recommendation_score']:.3f}",
            "key_strengths": [
                f"High validity ({assessment.get('validity', 0):.1%})",
                f"Measures relevant skills: {', '.join(assessment.get('skills_measured', [])[:3])}",
                f"Duration: {assessment.get('duration_minutes', 0)} minutes"
            ],
            "trade_offs": [
                "Balanced approach between performance and fairness",
                f"Fairness risk: {assessment.get('fairness_metrics', {}).get('adverse_impact_risk', 0):.1%}"
            ]
        }

if __name__ == "__main__":
    catalog_path = Path(__file__).parent.parent / "data" / "processed" / "assessment_catalog.json"
    
    engine = RAGEngine(str(catalog_path))
    
    result = engine.generate_recommendation(
        "Software engineer with strong analytical and problem-solving skills",
        {"fairness_weight": 0.5, "time_weight": 0.3}
    )
    
    print(json.dumps(result, indent=2))
