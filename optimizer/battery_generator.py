"""
Phase 4.1: Battery Enumeration with Search Space Control
Generates candidate assessment batteries with diversity constraints.
"""

from itertools import combinations
from typing import List, Dict, Set, Tuple
import json
from pathlib import Path

class BatteryGenerator:
    """Generates diverse candidate assessment batteries."""
    
    def __init__(self, assessment_catalog_path: str, max_battery_size: int = 4):
        """
        Initialize battery generator.
        
        Args:
            assessment_catalog_path: Path to assessment catalog JSON
            max_battery_size: Maximum number of assessments per battery
        """
        self.max_battery_size = max_battery_size
        self.assessments = self._load_assessments(assessment_catalog_path)
        self.cognitive_assessments = self._filter_by_type("cognitive")
        self.personality_assessments = self._filter_by_type("personality")
        self.situational_assessments = self._filter_by_type("situational")
        
    def _load_assessments(self, catalog_path: str) -> Dict:
        """Load assessment catalog."""
        with open(catalog_path, 'r') as f:
            catalog = json.load(f)
            # Convert list to dict for easier access
            assessments_dict = {}
            for assessment in catalog["assessments"]:
                aid = assessment["assessment_id"]
                assessments_dict[aid] = {
                    "type": assessment["assessment_type"],
                    "duration_minutes": assessment["duration_minutes"],
                    "primary_construct": list(assessment["measured_constructs"].keys())[0],
                    "constructs": assessment["measured_constructs"],
                    "reliability": assessment["reliability"],
                    "adverse_impact_risk": assessment["adverse_impact_risk"]
                }
            return assessments_dict
    
    def _filter_by_type(self, assessment_type: str) -> List[str]:
        """Filter assessments by type."""
        return [
            aid for aid, meta in self.assessments.items()
            if meta.get("type") == assessment_type
        ]
    
    def generate_candidate_batteries(self) -> List[Tuple[str, ...]]:
        """
        Generate all valid candidate batteries.
        
        Returns:
            List of battery tuples (assessment_ids)
        """
        candidates = []
        all_assessments = list(self.assessments.keys())
        
        # Generate batteries of size 1 to max_battery_size
        for size in range(1, self.max_battery_size + 1):
            for battery in combinations(all_assessments, size):
                if self._is_valid_battery(battery):
                    candidates.append(battery)
        
        return candidates
    
    def _is_valid_battery(self, battery: Tuple[str, ...]) -> bool:
        """
        Check if battery meets diversity constraints.
        
        Constraints:
        1. At least one cognitive assessment OR
        2. One personality + one cognitive assessment
        3. No redundant assessments (same construct)
        """
        battery_types = [self.assessments[aid]["type"] for aid in battery]
        battery_constructs = [self.assessments[aid]["primary_construct"] for aid in battery]
        
        # No duplicate constructs (diversity constraint)
        if len(set(battery_constructs)) != len(battery_constructs):
            return False
        
        # Must have at least one cognitive
        has_cognitive = "cognitive" in battery_types
        has_personality = "personality" in battery_types
        
        # Valid if: cognitive only, or personality + cognitive
        if has_cognitive:
            return True
        elif has_personality and len(battery) >= 2:
            # Personality-only batteries must be size >= 2
            return True
        
        return False
    
    def get_battery_metadata(self, battery: Tuple[str, ...]) -> Dict:
        """Get aggregated metadata for a battery."""
        total_duration = sum(
            self.assessments[aid]["duration_minutes"] for aid in battery
        )
        
        types = [self.assessments[aid]["type"] for aid in battery]
        constructs = [self.assessments[aid]["primary_construct"] for aid in battery]
        
        return {
            "battery_id": "_".join(sorted(battery)),
            "assessments": list(battery),
            "size": len(battery),
            "total_duration": total_duration,
            "types": types,
            "constructs": constructs,
            "has_cognitive": "cognitive" in types,
            "has_personality": "personality" in types,
            "has_situational": "situational" in types
        }
    
    def generate_batteries_with_metadata(self) -> List[Dict]:
        """Generate batteries with full metadata."""
        candidates = self.generate_candidate_batteries()
        return [self.get_battery_metadata(battery) for battery in candidates]


if __name__ == "__main__":
    # Test battery generation
    generator = BatteryGenerator("../data/processed/assessment_catalog.json")
    batteries = generator.generate_batteries_with_metadata()
    
    print(f"Generated {len(batteries)} candidate batteries")
    print(f"Size distribution: {[b['size'] for b in batteries[:10]]}")
    
    # Show examples
    for i, battery in enumerate(batteries[:5]):
        print(f"\nBattery {i+1}: {battery['battery_id']}")
        print(f"  Assessments: {battery['assessments']}")
        print(f"  Duration: {battery['total_duration']} min")
        print(f"  Types: {battery['types']}")