"""
Quick setup script for RAG system
Verifies all components are working
"""

import sys
from pathlib import Path
import json

def check_file_exists(filepath, description):
    """Check if file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} missing: {filepath}")
        return False

def test_imports():
    """Test if all modules can be imported."""
    print("\n" + "="*50)
    print("Testing Imports...")
    print("="*50)
    
    try:
        from rag.embeddings import VectorStore
        print("✅ RAG embeddings module")
    except ImportError as e:
        print(f"❌ RAG embeddings: {e}")
        return False
    
    try:
        from rag.rag_engine import RAGEngine
        print("✅ RAG engine module")
    except ImportError as e:
        print(f"❌ RAG engine: {e}")
        return False
    
    try:
        from scraper.shl_scraper import SHLCatalogScraper
        print("✅ Scraper module")
    except ImportError as e:
        print(f"❌ Scraper: {e}")
        return False
    
    try:
        from evaluation.metrics import RAGEvaluator
        print("✅ Evaluation module")
    except ImportError as e:
        print(f"❌ Evaluation: {e}")
        return False
    
    return True

def test_rag_engine():
    """Test RAG engine functionality."""
    print("\n" + "="*50)
    print("Testing RAG Engine...")
    print("="*50)
    
    try:
        from rag.rag_engine import RAGEngine
        
        catalog_path = Path(__file__).parent / "data" / "processed" / "assessment_catalog.json"
        
        if not catalog_path.exists():
            print(f"❌ Catalog not found: {catalog_path}")
            return False
        
        engine = RAGEngine(str(catalog_path))
        print("✅ RAG engine initialized")
        
        # Test recommendation
        result = engine.generate_recommendation(
            "Software engineer with analytical skills",
            {"fairness_weight": 0.5}
        )
        
        if result and "top_recommendation" in result:
            print("✅ Recommendation generated successfully")
            print(f"   Top: {result['top_recommendation'].get('name', 'Unknown')}")
            return True
        else:
            print("❌ Recommendation generation failed")
            return False
            
    except Exception as e:
        print(f"❌ RAG engine test failed: {e}")
        return False

def test_evaluation():
    """Test evaluation module."""
    print("\n" + "="*50)
    print("Testing Evaluation...")
    print("="*50)
    
    try:
        from evaluation.metrics import RAGEvaluator, create_test_dataset
        from rag.rag_engine import RAGEngine
        
        catalog_path = Path(__file__).parent / "data" / "processed" / "assessment_catalog.json"
        engine = RAGEngine(str(catalog_path))
        
        test_data = create_test_dataset()
        evaluator = RAGEvaluator()
        
        # Run evaluation
        metrics = evaluator.evaluate_end_to_end(test_data, engine)
        
        print("✅ Evaluation completed")
        print(f"   Overall Score: {metrics.get('overall_score', 0):.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False

def main():
    """Run all setup checks."""
    print("="*50)
    print("RAG SYSTEM SETUP VERIFICATION")
    print("="*50)
    
    # Check files
    print("\nChecking Files...")
    print("-"*50)
    
    files_ok = all([
        check_file_exists("data/processed/assessment_catalog.json", "Assessment catalog"),
        check_file_exists("rag/embeddings.py", "Embeddings module"),
        check_file_exists("rag/rag_engine.py", "RAG engine"),
        check_file_exists("scraper/shl_scraper.py", "Scraper"),
        check_file_exists("evaluation/metrics.py", "Evaluation"),
        check_file_exists("demo/simple_demo.py", "Web demo"),
        check_file_exists("docs/APPROACH_DOCUMENT.md", "Approach document"),
    ])
    
    # Test imports
    imports_ok = test_imports()
    
    # Test RAG engine
    rag_ok = test_rag_engine()
    
    # Test evaluation
    eval_ok = test_evaluation()
    
    # Summary
    print("\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)
    
    if files_ok and imports_ok and rag_ok and eval_ok:
        print("✅ All checks passed!")
        print("\nYour RAG system is ready!")
        print("\nNext steps:")
        print("1. Test locally: python demo/simple_demo.py")
        print("2. Run evaluation: python evaluation/metrics.py")
        print("3. Generate predictions: python evaluation/generate_predictions.py")
        print("4. Deploy: Follow GITHUB_DEPLOYMENT.md")
        print("5. Submit: Follow SUBMISSION_CHECKLIST.md")
        return True
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
