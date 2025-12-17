"""
Generate predictions for test dataset
Creates firstname_lastname.csv with predictions
"""

import csv
import json
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_engine import RAGEngine

def load_test_data(test_file: str) -> list:
    """Load test dataset."""
    # For demo, create sample test data
    return [
        {
            "id": 1,
            "job_description": "Software Engineer with strong analytical and problem-solving skills. Must be proficient in algorithms and data structures.",
            "required_skills": ["analytical", "problem-solving", "technical"]
        },
        {
            "id": 2,
            "job_description": "Sales Manager responsible for leading a team. Requires excellent communication and leadership abilities.",
            "required_skills": ["leadership", "communication", "team management"]
        },
        {
            "id": 3,
            "job_description": "Data Analyst with strong quantitative skills. Must be detail-oriented and proficient in statistical analysis.",
            "required_skills": ["analytical", "quantitative", "detail-oriented"]
        },
        {
            "id": 4,
            "job_description": "Customer Service Representative. Needs strong interpersonal skills and ability to handle stress.",
            "required_skills": ["communication", "resilience", "interpersonal"]
        },
        {
            "id": 5,
            "job_description": "Project Manager with organizational skills. Must be able to coordinate multiple tasks and lead teams.",
            "required_skills": ["organizational", "leadership", "multitasking"]
        },
        {
            "id": 6,
            "job_description": "Marketing Specialist with creative thinking. Requires strong communication and analytical skills.",
            "required_skills": ["creative", "communication", "analytical"]
        },
        {
            "id": 7,
            "job_description": "Financial Analyst with strong numerical reasoning. Must be detail-oriented and analytical.",
            "required_skills": ["numerical", "analytical", "detail-oriented"]
        },
        {
            "id": 8,
            "job_description": "HR Manager with people skills. Requires empathy, communication, and organizational abilities.",
            "required_skills": ["interpersonal", "communication", "organizational"]
        },
        {
            "id": 9,
            "job_description": "Operations Manager with process optimization skills. Needs analytical thinking and leadership.",
            "required_skills": ["analytical", "leadership", "process-oriented"]
        },
        {
            "id": 10,
            "job_description": "UX Designer with creative and technical skills. Must understand user needs and design principles.",
            "required_skills": ["creative", "technical", "user-focused"]
        }
    ]

def generate_predictions(output_file: str = "predictions.csv", firstname: str = "John", lastname: str = "Doe"):
    """Generate predictions and save to CSV."""
    
    # Initialize RAG engine
    catalog_path = Path(__file__).parent.parent / "data" / "processed" / "assessment_catalog.json"
    engine = RAGEngine(str(catalog_path))
    
    # Load test data
    test_data = load_test_data("test_data.json")
    
    # Generate predictions
    predictions = []
    for test_case in test_data:
        result = engine.generate_recommendation(
            query=test_case["job_description"],
            preferences={"fairness_weight": 0.5, "time_weight": 0.3}
        )
        
        top_rec = result["top_recommendation"]
        predictions.append({
            "id": test_case["id"],
            "job_description": test_case["job_description"],
            "predicted_assessment": top_rec["assessment_id"],
            "predicted_battery": top_rec.get("name", ""),
            "confidence_score": top_rec.get("recommendation_score", 0.0),
            "fairness_risk": top_rec.get("fairness_metrics", {}).get("adverse_impact_risk", 0.0),
            "duration_minutes": top_rec.get("duration_minutes", 0)
        })
    
    # Save to CSV
    output_path = Path(__file__).parent.parent / "predictions" / f"{firstname}_{lastname}.csv"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "job_description", "predicted_assessment", "predicted_battery",
            "confidence_score", "fairness_risk", "duration_minutes"
        ])
        writer.writeheader()
        writer.writerows(predictions)
    
    print(f"✅ Generated predictions for {len(predictions)} test cases")
    print(f"✅ Saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Update with your name
    generate_predictions(firstname="Kartik", lastname="Pandey")
