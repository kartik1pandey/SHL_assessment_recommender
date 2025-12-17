"""
GenAI Dataset Evaluation
Evaluate RAG system using company-provided train/test data
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Tuple
import re

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_engine import RAGEngine

class GenAIEvaluator:
    """Evaluator using company-provided GenAI dataset."""
    
    def __init__(self, catalog_path: str):
        self.rag_engine = RAGEngine(catalog_path)
        self.train_df = None
        self.test_df = None
        self.url_to_assessment_map = {}
        
    def load_datasets(self):
        """Load train and test datasets."""
        try:
            # Load from CSV files created by analyze_dataset.py
            self.train_df = pd.read_csv('data/processed/train_set.csv')
            self.test_df = pd.read_csv('data/processed/test_set.csv')
            
            print(f"✅ Loaded train set: {len(self.train_df)} samples")
            print(f"✅ Loaded test set: {len(self.test_df)} samples")
            
            # Create URL to assessment mapping
            self._create_url_mapping()
            
        except FileNotFoundError:
            print("❌ Dataset files not found. Run analyze_dataset.py first.")
            return False
        
        return True
    
    def _create_url_mapping(self):
        """Create mapping from URLs to assessment names."""
        # Extract assessment names from URLs
        for url in self.train_df['Assessment_url'].unique():
            assessment_name = self._extract_assessment_name_from_url(url)
            self.url_to_assessment_map[url] = assessment_name
        
        print(f"✅ Created mapping for {len(self.url_to_assessment_map)} unique assessments")
    
    def _extract_assessment_name_from_url(self, url: str) -> str:
        """Extract assessment name from SHL URL."""
        # Remove base URL and extract product name
        if 'product-catalog/view/' in url:
            product_part = url.split('product-catalog/view/')[-1]
            # Remove trailing slash and parameters
            product_name = product_part.rstrip('/').split('?')[0]
            # Convert to readable name
            return product_name.replace('-', ' ').title()
        else:
            return url.split('/')[-1].replace('-', ' ').title()
    
    def evaluate_on_test_set(self) -> Dict:
        """Evaluate RAG system on test set."""
        if self.test_df is None:
            print("❌ Test set not loaded")
            return {}
        
        print(f"\n🧪 EVALUATING ON TEST SET ({len(self.test_df)} queries)")
        print("=" * 60)
        
        results = []
        
        for idx, row in self.test_df.iterrows():
            query = row['Query']
            
            print(f"\n📝 Query {idx + 1}: {query[:80]}...")
            
            # Generate recommendation using RAG
            try:
                recommendation = self.rag_engine.generate_recommendation(
                    query=query,
                    preferences={"fairness_weight": 0.5, "time_weight": 0.3}
                )
                
                top_rec = recommendation.get("top_recommendation", {})
                alternatives = recommendation.get("alternatives", [])
                
                result = {
                    "query_id": idx + 1,
                    "query": query,
                    "top_recommendation": {
                        "assessment_id": top_rec.get("assessment_id", ""),
                        "name": top_rec.get("name", ""),
                        "score": top_rec.get("recommendation_score", 0.0)
                    },
                    "alternatives": [
                        {
                            "assessment_id": alt.get("assessment_id", ""),
                            "name": alt.get("name", ""),
                            "score": alt.get("recommendation_score", 0.0)
                        }
                        for alt in alternatives[:3]
                    ]
                }
                
                results.append(result)
                
                print(f"   ✅ Top: {top_rec.get('name', 'Unknown')} (Score: {top_rec.get('recommendation_score', 0):.3f})")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                results.append({
                    "query_id": idx + 1,
                    "query": query,
                    "error": str(e)
                })
        
        return {
            "test_results": results,
            "total_queries": len(self.test_df),
            "successful_predictions": len([r for r in results if "error" not in r])
        }
    
    def analyze_training_patterns(self) -> Dict:
        """Analyze patterns in training data."""
        if self.train_df is None:
            print("❌ Training set not loaded")
            return {}
        
        print(f"\n📊 ANALYZING TRAINING PATTERNS")
        print("=" * 60)
        
        # Group by query to see multiple assessments per query
        query_groups = self.train_df.groupby('Query')
        
        analysis = {
            "unique_queries": len(query_groups),
            "total_samples": len(self.train_df),
            "avg_assessments_per_query": len(self.train_df) / len(query_groups),
            "query_patterns": []
        }
        
        print(f"📈 Training Data Statistics:")
        print(f"   Unique queries: {analysis['unique_queries']}")
        print(f"   Total samples: {analysis['total_samples']}")
        print(f"   Avg assessments per query: {analysis['avg_assessments_per_query']:.1f}")
        
        # Analyze each query pattern
        for query, group in query_groups:
            assessments = [self.url_to_assessment_map.get(url, url) for url in group['Assessment_url']]
            
            pattern = {
                "query": query[:100] + "..." if len(query) > 100 else query,
                "num_assessments": len(assessments),
                "assessments": assessments
            }
            
            analysis["query_patterns"].append(pattern)
            
            print(f"\n📝 Query: {query[:60]}...")
            print(f"   Assessments ({len(assessments)}):")
            for i, assessment in enumerate(assessments[:5]):  # Show first 5
                print(f"     {i+1}. {assessment}")
            if len(assessments) > 5:
                print(f"     ... and {len(assessments) - 5} more")
        
        return analysis
    
    def generate_test_predictions(self, output_file: str = "predictions/Kartik_Pandey_GenAI.csv"):
        """Generate predictions for test set in required format."""
        if self.test_df is None:
            print("❌ Test set not loaded")
            return
        
        print(f"\n📊 GENERATING TEST PREDICTIONS")
        print("=" * 60)
        
        predictions = []
        
        for idx, row in self.test_df.iterrows():
            query = row['Query']
            
            try:
                # Generate recommendation
                recommendation = self.rag_engine.generate_recommendation(
                    query=query,
                    preferences={"fairness_weight": 0.5, "time_weight": 0.3}
                )
                
                top_rec = recommendation.get("top_recommendation", {})
                
                prediction = {
                    "id": idx + 1,
                    "query": query,
                    "predicted_assessment_id": top_rec.get("assessment_id", ""),
                    "predicted_assessment_name": top_rec.get("name", ""),
                    "confidence_score": top_rec.get("recommendation_score", 0.0),
                    "method": "RAG",
                    "fairness_risk": top_rec.get("fairness_metrics", {}).get("adverse_impact_risk", 0.0),
                    "duration_minutes": top_rec.get("duration_minutes", 0)
                }
                
                predictions.append(prediction)
                
                print(f"✅ Query {idx + 1}: {top_rec.get('name', 'Unknown')}")
                
            except Exception as e:
                print(f"❌ Query {idx + 1}: Error - {str(e)}")
                predictions.append({
                    "id": idx + 1,
                    "query": query,
                    "predicted_assessment_id": "ERROR",
                    "predicted_assessment_name": f"Error: {str(e)}",
                    "confidence_score": 0.0,
                    "method": "RAG",
                    "fairness_risk": 0.0,
                    "duration_minutes": 0
                })
        
        # Save predictions
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Saved {len(predictions)} predictions to: {output_path}")
        return output_path
    
    def run_complete_evaluation(self):
        """Run complete evaluation pipeline."""
        print("🎯 GENAI DATASET EVALUATION")
        print("=" * 80)
        
        # Load datasets
        if not self.load_datasets():
            return
        
        # Analyze training patterns
        training_analysis = self.analyze_training_patterns()
        
        # Evaluate on test set
        test_results = self.evaluate_on_test_set()
        
        # Generate predictions
        pred_file = self.generate_test_predictions()
        
        # Summary
        print(f"\n🎉 EVALUATION COMPLETE")
        print("=" * 80)
        print(f"✅ Training queries analyzed: {training_analysis.get('unique_queries', 0)}")
        print(f"✅ Test queries evaluated: {test_results.get('total_queries', 0)}")
        print(f"✅ Successful predictions: {test_results.get('successful_predictions', 0)}")
        print(f"✅ Predictions saved to: {pred_file}")
        
        return {
            "training_analysis": training_analysis,
            "test_results": test_results,
            "predictions_file": str(pred_file)
        }

def main():
    """Main evaluation function."""
    # Initialize evaluator
    catalog_path = Path(__file__).parent.parent / "data" / "processed" / "assessment_catalog.json"
    evaluator = GenAIEvaluator(str(catalog_path))
    
    # Run complete evaluation
    results = evaluator.run_complete_evaluation()
    
    # Save detailed results
    with open("evaluation/genai_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: evaluation/genai_evaluation_results.json")

if __name__ == "__main__":
    main()