"""
Build and manage assessment catalog from raw data sources.
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
from .schema import Assessment, AssessmentCategory, DifficultyLevel


class AssessmentCatalogBuilder:
    """Builds assessment catalog from various data sources."""
    
    def __init__(self, raw_data_path: str = "data/raw"):
        self.raw_data_path = Path(raw_data_path)
        self.assessments = []
    
    def load_shl_catalog(self, filename: str = "shl_catalog_raw.csv") -> List[Assessment]:
        """Load SHL assessment catalog from CSV."""
        csv_path = self.raw_data_path / filename
        
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found")
            return []
        
        df = pd.read_csv(csv_path)
        assessments = []
        
        for _, row in df.iterrows():
            try:
                assessment = Assessment(
                    id=row['assessment_id'],
                    name=row['assessment_name'],
                    category=AssessmentCategory(row['category'].lower()),
                    skills_measured=row['skills_measured'].split(';'),
                    duration_minutes=int(row['duration_minutes']),
                    difficulty=DifficultyLevel(row['difficulty_level'].lower()),
                    description=row['description'],
                    job_relevance=[],  # Will be populated later
                    provider="SHL"
                )
                assessments.append(assessment)
            except Exception as e:
                print(f"Error processing row {row['assessment_id']}: {e}")
        
        return assessments
    
    def add_job_relevance_mapping(self, assessments: List[Assessment]) -> List[Assessment]:
        """Add job relevance mapping based on skills."""
        
        # Simple mapping based on assessment categories and skills
        job_mappings = {
            "programming": ["software_engineer", "developer", "programmer"],
            "sql": ["data_analyst", "database_administrator"],
            "sales": ["sales_executive", "account_manager"],
            "numerical": ["data_analyst", "financial_analyst"],
            "leadership": ["manager", "team_lead", "director"]
        }
        
        for assessment in assessments:
            relevance = set()
            
            # Map based on skills measured
            for skill in assessment.skills_measured:
                for key, jobs in job_mappings.items():
                    if key.lower() in skill.lower():
                        relevance.update(jobs)
            
            # Map based on assessment name
            name_lower = assessment.name.lower()
            if "programming" in name_lower or "code" in name_lower:
                relevance.update(job_mappings["programming"])
            elif "sql" in name_lower or "database" in name_lower:
                relevance.update(job_mappings["sql"])
            elif "sales" in name_lower:
                relevance.update(job_mappings["sales"])
            elif "numerical" in name_lower or "math" in name_lower:
                relevance.update(job_mappings["numerical"])
            
            assessment.job_relevance = list(relevance)
        
        return assessments
    
    def build_catalog(self) -> List[Assessment]:
        """Build complete assessment catalog."""
        # Load from different sources
        shl_assessments = self.load_shl_catalog()
        
        # Combine all assessments
        all_assessments = shl_assessments
        
        # Add job relevance mapping
        all_assessments = self.add_job_relevance_mapping(all_assessments)
        
        self.assessments = all_assessments
        return all_assessments
    
    def save_catalog(self, output_path: str = "data/processed/assessment_catalog.json"):
        """Save catalog to JSON file."""
        catalog_data = {
            "assessments": [assessment.to_dict() for assessment in self.assessments],
            "metadata": {
                "total_assessments": len(self.assessments),
                "categories": list(set(a.category.value for a in self.assessments)),
                "providers": list(set(a.provider for a in self.assessments if a.provider))
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(catalog_data, f, indent=2)
        
        print(f"Catalog saved to {output_path}")
    
    def get_assessments_by_category(self, category: AssessmentCategory) -> List[Assessment]:
        """Filter assessments by category."""
        return [a for a in self.assessments if a.category == category]
    
    def get_assessments_by_job(self, job_title: str) -> List[Assessment]:
        """Get assessments relevant for a specific job."""
        job_lower = job_title.lower().replace(" ", "_")
        return [a for a in self.assessments if job_lower in a.job_relevance]


if __name__ == "__main__":
    builder = AssessmentCatalogBuilder()
    assessments = builder.build_catalog()
    builder.save_catalog()
    
    print(f"Built catalog with {len(assessments)} assessments")
    for category in AssessmentCategory:
        count = len(builder.get_assessments_by_category(category))
        print(f"  {category.value}: {count} assessments")