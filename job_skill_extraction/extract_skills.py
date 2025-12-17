"""
Extract skills from job descriptions using NLP techniques.
"""

import re
import json
from typing import List, Dict, Set, Tuple
from pathlib import Path
from collections import Counter
import spacy
from dataclasses import dataclass


@dataclass
class ExtractedSkill:
    """Represents an extracted skill with metadata."""
    skill: str
    confidence: float
    context: str
    extraction_method: str


class SkillExtractor:
    """Extracts skills from job descriptions."""
    
    def __init__(self, skills_ontology_path: str = "ontology/skills.json"):
        self.skills_ontology = self._load_skills_ontology(skills_ontology_path)
        self.skill_patterns = self._build_skill_patterns()
        
        # Try to load spaCy model, fallback to basic extraction if not available
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except OSError:
            print("Warning: spaCy model not found. Using basic pattern matching.")
            self.nlp = None
            self.use_spacy = False
    
    def _load_skills_ontology(self, path: str) -> Dict:
        """Load skills ontology from JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Skills ontology not found at {path}")
            return {}
    
    def _build_skill_patterns(self) -> List[str]:
        """Build regex patterns for skill extraction."""
        patterns = []
        
        # Get all skills from ontology
        all_skills = set()
        
        if "skill_hierarchy" in self.skills_ontology:
            def extract_skills_recursive(obj):
                if isinstance(obj, dict):
                    for value in obj.values():
                        extract_skills_recursive(value)
                elif isinstance(obj, list):
                    all_skills.update(obj)
            
            extract_skills_recursive(self.skills_ontology["skill_hierarchy"])
        
        # Add synonyms
        if "skill_synonyms" in self.skills_ontology:
            for skill, synonyms in self.skills_ontology["skill_synonyms"].items():
                all_skills.add(skill)
                all_skills.update(synonyms)
        
        # Convert to regex patterns
        for skill in all_skills:
            # Handle multi-word skills
            pattern = re.escape(skill.replace("_", " "))
            patterns.append(pattern)
        
        return patterns
    
    def extract_skills_pattern_matching(self, text: str) -> List[ExtractedSkill]:
        """Extract skills using pattern matching."""
        extracted = []
        text_lower = text.lower()
        
        for pattern in self.skill_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                skill = match.group().replace(" ", "_")
                context = self._get_context(text, match.start(), match.end())
                
                extracted.append(ExtractedSkill(
                    skill=skill,
                    confidence=0.8,  # High confidence for exact matches
                    context=context,
                    extraction_method="pattern_matching"
                ))
        
        return extracted
    
    def extract_skills_spacy(self, text: str) -> List[ExtractedSkill]:
        """Extract skills using spaCy NLP."""
        if not self.use_spacy:
            return []
        
        doc = self.nlp(text)
        extracted = []
        
        # Extract based on named entities and noun phrases
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "LANGUAGE"]:
                skill = ent.text.lower().replace(" ", "_")
                if self._is_valid_skill(skill):
                    extracted.append(ExtractedSkill(
                        skill=skill,
                        confidence=0.6,
                        context=ent.sent.text,
                        extraction_method="spacy_ner"
                    ))
        
        # Extract noun phrases that might be skills
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to reasonable skill names
                skill = chunk.text.lower().replace(" ", "_")
                if self._is_valid_skill(skill):
                    extracted.append(ExtractedSkill(
                        skill=skill,
                        confidence=0.4,
                        context=chunk.sent.text,
                        extraction_method="spacy_noun_phrases"
                    ))
        
        return extracted
    
    def extract_skills_keywords(self, text: str) -> List[ExtractedSkill]:
        """Extract skills based on common technical keywords."""
        keywords = [
            "python", "javascript", "java", "sql", "react", "angular", "vue",
            "machine learning", "data analysis", "statistics", "tableau",
            "aws", "azure", "docker", "kubernetes", "git", "agile", "scrum",
            "communication", "leadership", "problem solving", "teamwork"
        ]
        
        extracted = []
        text_lower = text.lower()
        
        for keyword in keywords:
            if keyword in text_lower:
                # Find the context
                start_idx = text_lower.find(keyword)
                context = self._get_context(text, start_idx, start_idx + len(keyword))
                
                extracted.append(ExtractedSkill(
                    skill=keyword.replace(" ", "_"),
                    confidence=0.7,
                    context=context,
                    extraction_method="keyword_matching"
                ))
        
        return extracted
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context around a matched skill."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _is_valid_skill(self, skill: str) -> bool:
        """Check if extracted text is likely a valid skill."""
        # Basic validation rules
        if len(skill) < 2 or len(skill) > 50:
            return False
        
        # Skip common words that aren't skills
        common_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        if skill in common_words:
            return False
        
        return True
    
    def extract_skills(self, text: str) -> List[ExtractedSkill]:
        """Extract skills using all available methods."""
        all_extracted = []
        
        # Pattern matching (highest priority)
        all_extracted.extend(self.extract_skills_pattern_matching(text))
        
        # spaCy extraction
        all_extracted.extend(self.extract_skills_spacy(text))
        
        # Keyword extraction
        all_extracted.extend(self.extract_skills_keywords(text))
        
        # Deduplicate and rank by confidence
        return self._deduplicate_skills(all_extracted)
    
    def _deduplicate_skills(self, skills: List[ExtractedSkill]) -> List[ExtractedSkill]:
        """Remove duplicates and keep highest confidence version."""
        skill_map = {}
        
        for skill in skills:
            if skill.skill not in skill_map or skill.confidence > skill_map[skill.skill].confidence:
                skill_map[skill.skill] = skill
        
        return list(skill_map.values())
    
    def extract_from_job_description(self, job_description: str) -> Dict[str, any]:
        """Extract skills from a job description and return structured result."""
        extracted_skills = self.extract_skills(job_description)
        
        # Categorize skills
        categorized = {
            "technical": [],
            "cognitive": [],
            "behavioral": []
        }
        
        for skill in extracted_skills:
            category = self._categorize_skill(skill.skill)
            categorized[category].append({
                "skill": skill.skill,
                "confidence": skill.confidence,
                "context": skill.context,
                "method": skill.extraction_method
            })
        
        return {
            "total_skills": len(extracted_skills),
            "skills_by_category": categorized,
            "all_skills": [s.skill for s in extracted_skills]
        }
    
    def _categorize_skill(self, skill: str) -> str:
        """Categorize a skill into technical, cognitive, or behavioral."""
        technical_indicators = ["programming", "database", "cloud", "framework", "language"]
        cognitive_indicators = ["analysis", "reasoning", "problem", "thinking"]
        
        skill_lower = skill.lower()
        
        for indicator in technical_indicators:
            if indicator in skill_lower:
                return "technical"
        
        for indicator in cognitive_indicators:
            if indicator in skill_lower:
                return "cognitive"
        
        return "behavioral"  # Default category


def extract_skills_from_file(file_path: str, extractor: SkillExtractor = None) -> Dict:
    """Extract skills from a job description file."""
    if extractor is None:
        extractor = SkillExtractor()
    
    with open(file_path, 'r') as f:
        job_description = f.read()
    
    return extractor.extract_from_job_description(job_description)


if __name__ == "__main__":
    # Example usage
    extractor = SkillExtractor()
    
    # Process sample job descriptions
    job_files = [
        "data/raw/job_descriptions/jd_software_engineer.txt",
        "data/raw/job_descriptions/jd_data_analyst.txt",
        "data/raw/job_descriptions/jd_sales_exec.txt"
    ]
    
    for job_file in job_files:
        if Path(job_file).exists():
            print(f"\nProcessing: {job_file}")
            result = extract_skills_from_file(job_file, extractor)
            print(f"Found {result['total_skills']} skills")
            
            for category, skills in result['skills_by_category'].items():
                if skills:
                    print(f"  {category}: {[s['skill'] for s in skills[:5]]}")  # Show first 5