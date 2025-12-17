"""
Skills ontology and taxonomy management.
"""

from typing import Dict, List, Set
import json
from pathlib import Path


class SkillsOntology:
    """Manages the skills taxonomy and relationships."""
    
    def __init__(self, skills_file: str = None):
        self.skills_file = skills_file or "skills.json"
        self.skills_data = self._load_skills()
        
    def _load_skills(self) -> Dict:
        """Load skills taxonomy from JSON file."""
        try:
            with open(self.skills_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_skills()
    
    def _default_skills(self) -> Dict:
        """Return default skills structure."""
        return {
            "technical_skills": {},
            "cognitive_skills": [],
            "soft_skills": {}
        }
    
    def get_all_skills(self) -> Set[str]:
        """Get all skills as a flat set."""
        skills = set()
        
        # Technical skills
        for category, skill_list in self.skills_data.get("technical_skills", {}).items():
            skills.update(skill_list)
        
        # Cognitive skills
        skills.update(self.skills_data.get("cognitive_skills", []))
        
        # Soft skills
        for category, skill_list in self.skills_data.get("soft_skills", {}).items():
            skills.update(skill_list)
            
        return skills
    
    def get_skills_by_category(self, category: str) -> List[str]:
        """Get skills for a specific category."""
        if category in self.skills_data:
            return self.skills_data[category]
        return []
    
    def find_related_skills(self, skill: str) -> List[str]:
        """Find skills related to the given skill."""
        # Simple implementation - can be enhanced with semantic similarity
        related = []
        all_skills = self.get_all_skills()
        
        for s in all_skills:
            if skill.lower() in s.lower() or s.lower() in skill.lower():
                if s != skill:
                    related.append(s)
        
        return related