"""
Evaluation Metrics for RAG System
Measures accuracy and effectiveness at key stages
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

class RAGEvaluator:
    """Evaluator for RAG-based recommendation system."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_retrieval(self, queries: List[str], retrieved: List[List[Dict]], 
                          ground_truth: List[List[str]]) -> Dict:
        """
        Evaluate retrieval stage.
        
        Metrics:
        - Precision@K
        - Recall@K
        - MRR (Mean Reciprocal Rank)
        - NDCG (Normalized Discounted Cumulative Gain)
        """
        precisions = []
        recalls = []
        mrrs = []
        
        for ret, truth in zip(retrieved, ground_truth):
            ret_ids = [r["assessment"]["assessment_id"] for r in ret]
            
            # Precision@K
            relevant = len(set(ret_ids) & set(truth))
            precision = relevant / len(ret_ids) if ret_ids else 0
            precisions.append(precision)
            
            # Recall@K
            recall = relevant / len(truth) if truth else 0
            recalls.append(recall)
            
            # MRR
            for i, rid in enumerate(ret_ids):
                if rid in truth:
                    mrrs.append(1.0 / (i + 1))
                    break
            else:
                mrrs.append(0.0)
        
        return {
            "precision@5": np.mean(precisions),
            "recall@5": np.mean(recalls),
            "mrr": np.mean(mrrs),
            "num_queries": len(queries)
        }
    
    def evaluate_ranking(self, recommendations: List[Dict], 
                        ground_truth: List[Dict]) -> Dict:
        """
        Evaluate ranking quality.
        
        Metrics:
        - NDCG
        - Spearman correlation
        - Top-1 accuracy
        """
        top1_correct = 0
        ndcgs = []
        
        for rec, truth in zip(recommendations, ground_truth):
            # Top-1 accuracy
            if rec["top_recommendation"]["assessment_id"] == truth["best_assessment"]:
                top1_correct += 1
            
            # NDCG (simplified)
            rec_ids = [rec["top_recommendation"]["assessment_id"]]
            rec_ids += [alt["assessment_id"] for alt in rec.get("alternatives", [])]
            
            dcg = 0
            for i, rid in enumerate(rec_ids):
                relevance = 1 if rid == truth["best_assessment"] else 0.5 if rid in truth.get("acceptable", []) else 0
                dcg += relevance / np.log2(i + 2)
            
            # Ideal DCG
            ideal_relevances = sorted([1] + [0.5] * len(truth.get("acceptable", [])), reverse=True)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
            
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)
        
        return {
            "top1_accuracy": top1_correct / len(recommendations),
            "ndcg": np.mean(ndcgs),
            "num_recommendations": len(recommendations)
        }
    
    def evaluate_fairness(self, recommendations: List[Dict]) -> Dict:
        """
        Evaluate fairness of recommendations.
        
        Metrics:
        - Average adverse impact risk
        - Fairness-performance trade-off
        """
        fairness_risks = []
        validities = []
        
        for rec in recommendations:
            top = rec["top_recommendation"]
            fairness_metrics = top.get("fairness_metrics", {})
            fairness_risks.append(fairness_metrics.get("adverse_impact_risk", 0.5))
            validities.append(top.get("validity", 0.5))
        
        return {
            "avg_fairness_risk": np.mean(fairness_risks),
            "avg_validity": np.mean(validities),
            "fairness_validity_correlation": np.corrcoef(fairness_risks, validities)[0, 1]
        }
    
    def evaluate_end_to_end(self, test_data: List[Dict], 
                           rag_engine) -> Dict:
        """
        End-to-end evaluation.
        
        Args:
            test_data: List of test cases with queries and ground truth
            rag_engine: RAG engine to evaluate
        
        Returns:
            Comprehensive metrics
        """
        queries = [t["query"] for t in test_data]
        ground_truth_retrieval = [t.get("relevant_assessments", []) for t in test_data]
        ground_truth_ranking = [t.get("best_recommendation", {}) for t in test_data]
        
        # Generate recommendations
        recommendations = []
        retrieved_results = []
        
        for query in queries:
            result = rag_engine.generate_recommendation(query)
            recommendations.append(result)
            retrieved_results.append(result.get("retrieval_results", []))
        
        # Evaluate stages
        retrieval_metrics = self.evaluate_retrieval(queries, retrieved_results, ground_truth_retrieval)
        ranking_metrics = self.evaluate_ranking(recommendations, ground_truth_ranking)
        fairness_metrics = self.evaluate_fairness(recommendations)
        
        return {
            "retrieval": retrieval_metrics,
            "ranking": ranking_metrics,
            "fairness": fairness_metrics,
            "overall_score": (
                retrieval_metrics["precision@5"] * 0.3 +
                ranking_metrics["top1_accuracy"] * 0.4 +
                (1 - fairness_metrics["avg_fairness_risk"]) * 0.3
            )
        }

def create_test_dataset() -> List[Dict]:
    """Create test dataset for evaluation."""
    return [
        {
            "query": "Software engineer with strong analytical skills",
            "relevant_assessments": ["SHL_NUM_01", "SHL_VER_01", "SHL_OPQ_01"],
            "best_recommendation": {
                "best_assessment": "SHL_NUM_01",
                "acceptable": ["SHL_VER_01", "SHL_OPQ_01"]
            }
        },
        {
            "query": "Sales manager with leadership and communication skills",
            "relevant_assessments": ["SHL_OPQ_01", "SHL_SJT_01", "SHL_MQ_01"],
            "best_recommendation": {
                "best_assessment": "SHL_OPQ_01",
                "acceptable": ["SHL_SJT_01", "SHL_MQ_01"]
            }
        },
        {
            "query": "Data analyst with quantitative and detail-oriented skills",
            "relevant_assessments": ["SHL_NUM_01", "SHL_VER_01", "SHL_OPQ_01"],
            "best_recommendation": {
                "best_assessment": "SHL_NUM_01",
                "acceptable": ["SHL_VER_01"]
            }
        }
    ]

if __name__ == "__main__":
    from rag.rag_engine import RAGEngine
    
    # Load RAG engine
    catalog_path = Path(__file__).parent.parent / "data" / "processed" / "assessment_catalog.json"
    engine = RAGEngine(str(catalog_path))
    
    # Create test data
    test_data = create_test_dataset()
    
    # Evaluate
    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate_end_to_end(test_data, engine)
    
    print("=" * 50)
    print("RAG SYSTEM EVALUATION RESULTS")
    print("=" * 50)
    print(json.dumps(metrics, indent=2))
