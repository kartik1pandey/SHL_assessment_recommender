"""
Phase 4.3: Fairness Penalty Modeling
Computes fairness risk for assessment batteries based on adverse impact potential.
"""

import math
from typing import Dict, List, Tuple
import json

class FairnessAnalyzer:
    """Analyzes and penalizes assessment batteries for fairness risk."""
    
    def __init__(self, assessment_catalog_path: str):
        """
        Initialize fairness analyzer.
        
        Args:
            assessment_catalog_path: Path to assessment catalog with fairness metadata
        """
        self.assessments = self._load_assessments(assessment_catalog_path)
        self.adverse_impact_weights = self._initialize_adverse_impact_weights()
        
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
                    "adverse_impact_risk": assessment["adverse_impact_risk"],
                    "cultural_loading": "medium"  # Default value
                }
            return assessments_dict
    
    def _initialize_adverse_impact_weights(self) -> Dict[str, float]:
        """
        Initialize adverse impact weights for different assessment types.
        
        Based on research literature on group differences in assessment performance.
        Higher values = higher adverse impact risk.
        """
        return {
            # Cognitive assessments (higher adverse impact risk)
            "cognitive": 0.8,
            
            # Personality assessments (moderate risk, depends on construct)
            "personality": 0.3,
            
            # Situational judgment (lower risk, more job-relevant)
            "situational": 0.2,
            
            # Work samples (lowest risk)
            "work_sample": 0.1
        }
    
    def compute_battery_fairness_risk(self, battery_metadata: Dict) -> Dict:
        """
        Compute fairness risk for an assessment battery.
        
        Args:
            battery_metadata: Battery metadata from BatteryGenerator
            
        Returns:
            Fairness risk metrics
        """
        assessments = battery_metadata["assessments"]
        
        # Individual assessment risks
        individual_risks = []
        for assessment_id in assessments:
            risk = self._compute_assessment_fairness_risk(assessment_id)
            individual_risks.append(risk)
        
        # Aggregate battery risk
        battery_risk = self._aggregate_fairness_risk(individual_risks, assessments)
        
        # Risk breakdown by category
        risk_breakdown = self._compute_risk_breakdown(assessments)
        
        return {
            "total_fairness_risk": battery_risk,
            "individual_risks": dict(zip(assessments, individual_risks)),
            "risk_breakdown": risk_breakdown,
            "high_risk_assessments": [
                aid for aid, risk in zip(assessments, individual_risks) 
                if risk > 0.6
            ]
        }
    
    def _compute_assessment_fairness_risk(self, assessment_id: str) -> float:
        """
        Compute fairness risk for individual assessment.
        
        Risk factors:
        1. Assessment type (cognitive > personality > situational)
        2. Construct measured (fluid intelligence > crystallized > personality)
        3. Cultural loading (language-heavy > non-verbal)
        """
        if assessment_id not in self.assessments:
            return 0.5  # Default moderate risk
        
        assessment = self.assessments[assessment_id]
        
        # Base risk from assessment type
        assessment_type = assessment.get("type", "unknown")
        base_risk = self.adverse_impact_weights.get(assessment_type, 0.5)
        
        # Adjust for specific constructs
        construct = assessment.get("primary_construct", "")
        construct_adjustment = self._get_construct_risk_adjustment(construct)
        
        # Adjust for cultural loading
        cultural_loading = assessment.get("cultural_loading", "medium")
        cultural_adjustment = self._get_cultural_loading_adjustment(cultural_loading)
        
        # Combine risk factors (multiplicative model)
        total_risk = base_risk * construct_adjustment * cultural_adjustment
        
        return min(total_risk, 1.0)  # Cap at 1.0
    
    def _get_construct_risk_adjustment(self, construct: str) -> float:
        """Get risk adjustment based on construct type."""
        construct_risks = {
            # Cognitive constructs (higher risk)
            "C1": 1.2,  # Fluid intelligence
            "C2": 1.1,  # Processing speed
            "C3": 1.0,  # Verbal ability
            "C4": 0.9,  # Spatial ability
            "C5": 0.8,  # Memory
            
            # Behavioral constructs (moderate risk)
            "B1": 0.7,  # Conscientiousness
            "B2": 0.8,  # Extraversion
            "B3": 0.6,  # Agreeableness
            "B4": 0.7,  # Emotional stability
            "B5": 0.8,  # Openness
            
            # Work-style constructs (lower risk)
            "W1": 0.5,  # Teamwork
            "W2": 0.6,  # Leadership
            "W3": 0.5,  # Adaptability
            "W4": 0.6,  # Communication
            "W5": 0.5   # Initiative
        }
        
        return construct_risks.get(construct, 1.0)
    
    def _get_cultural_loading_adjustment(self, cultural_loading: str) -> float:
        """Get risk adjustment based on cultural loading."""
        loading_adjustments = {
            "high": 1.3,    # Language-heavy, culturally specific
            "medium": 1.0,  # Some cultural elements
            "low": 0.8      # Culture-fair, non-verbal
        }
        
        return loading_adjustments.get(cultural_loading, 1.0)
    
    def _aggregate_fairness_risk(self, individual_risks: List[float], assessments: List[str]) -> float:
        """
        Aggregate individual assessment risks into battery risk.
        
        Uses weighted average with penalty for overlapping high-risk assessments.
        """
        if not individual_risks:
            return 0.0
        
        # Base risk: weighted average
        base_risk = sum(individual_risks) / len(individual_risks) if individual_risks else 0.0
        
        # Penalty for multiple high-risk assessments
        high_risk_count = sum(1 for risk in individual_risks if risk > 0.6)
        overlap_penalty = 0.1 * max(0, high_risk_count - 1)
        
        # Penalty for cognitive-heavy batteries
        cognitive_count = sum(
            1 for aid in assessments 
            if self.assessments.get(aid, {}).get("type") == "cognitive"
        )
        cognitive_penalty = 0.15 * max(0, cognitive_count - 2)
        
        total_risk = base_risk + overlap_penalty + cognitive_penalty
        
        return min(total_risk, 1.0)
    
    def _compute_risk_breakdown(self, assessments: List[str]) -> Dict[str, float]:
        """Compute risk breakdown by assessment type."""
        type_risks = {}
        type_counts = {}
        
        for assessment_id in assessments:
            if assessment_id in self.assessments:
                assessment_type = self.assessments[assessment_id].get("type", "unknown")
                risk = self._compute_assessment_fairness_risk(assessment_id)
                
                if assessment_type not in type_risks:
                    type_risks[assessment_type] = 0.0
                    type_counts[assessment_type] = 0
                
                type_risks[assessment_type] += risk
                type_counts[assessment_type] += 1
        
        # Average risk by type
        breakdown = {}
        for assessment_type in type_risks:
            breakdown[assessment_type] = type_risks[assessment_type] / type_counts[assessment_type]
        
        return breakdown
    
    def compare_batteries_fairness(self, batteries_metadata: List[Dict]) -> List[Dict]:
        """
        Compare batteries on fairness metrics.
        
        Returns:
            Batteries sorted by fairness risk (ascending)
        """
        results = []
        
        for battery_meta in batteries_metadata:
            fairness_analysis = self.compute_battery_fairness_risk(battery_meta)
            
            result = {
                **battery_meta,
                **fairness_analysis
            }
            results.append(result)
        
        # Sort by fairness risk (ascending - lower risk is better)
        results.sort(key=lambda x: x["total_fairness_risk"])
        
        return results
    
    def get_fairness_recommendations(self, battery_metadata: Dict) -> List[str]:
        """
        Get recommendations to reduce fairness risk.
        
        Returns:
            List of actionable recommendations
        """
        fairness_analysis = self.compute_battery_fairness_risk(battery_metadata)
        recommendations = []
        
        total_risk = fairness_analysis["total_fairness_risk"]
        high_risk_assessments = fairness_analysis["high_risk_assessments"]
        
        if total_risk > 0.7:
            recommendations.append("HIGH RISK: Consider reducing cognitive assessments")
        
        if len(high_risk_assessments) > 1:
            recommendations.append(f"Remove high-risk assessments: {', '.join(high_risk_assessments)}")
        
        cognitive_count = sum(
            1 for aid in battery_metadata["assessments"]
            if self.assessments.get(aid, {}).get("type") == "cognitive"
        )
        
        if cognitive_count > 2:
            recommendations.append("Consider replacing cognitive tests with situational judgment")
        
        if not recommendations:
            recommendations.append("Fairness risk is acceptable")
        
        return recommendations


if __name__ == "__main__":
    # Test fairness analysis
    analyzer = FairnessAnalyzer("../data/processed/assessment_catalog.json")
    
    # Test high-risk battery (cognitive-heavy)
    high_risk_battery = {
        "battery_id": "SHL_NUM_01_SHL_VER_01_SHL_LOG_01",
        "assessments": ["SHL_NUM_01", "SHL_VER_01", "SHL_LOG_01"],
        "size": 3,
        "total_duration": 90
    }
    
    # Test low-risk battery (mixed)
    low_risk_battery = {
        "battery_id": "OPQ_BHV_SJT_GEN",
        "assessments": ["OPQ_BHV", "SJT_GEN"],
        "size": 2,
        "total_duration": 60
    }
    
    print("High-Risk Battery Analysis:")
    high_risk_analysis = analyzer.compute_battery_fairness_risk(high_risk_battery)
    print(f"  Total Risk: {high_risk_analysis['total_fairness_risk']:.3f}")
    print(f"  High-Risk Assessments: {high_risk_analysis['high_risk_assessments']}")
    
    print("\nLow-Risk Battery Analysis:")
    low_risk_analysis = analyzer.compute_battery_fairness_risk(low_risk_battery)
    print(f"  Total Risk: {low_risk_analysis['total_fairness_risk']:.3f}")
    print(f"  High-Risk Assessments: {low_risk_analysis['high_risk_assessments']}")
    
    print("\nRecommendations for High-Risk Battery:")
    recommendations = analyzer.get_fairness_recommendations(high_risk_battery)
    for rec in recommendations:
        print(f"  - {rec}")