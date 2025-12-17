"""
Phase 5.1: Decision Trace Generation
Generates human-readable explanations for assessment battery recommendations.
"""

from typing import Dict, List, Tuple, Any
import json

class DecisionTracer:
    """Generates explainable decision traces for assessment battery recommendations."""
    
    def __init__(self, assessment_catalog_path: str):
        """Initialize with assessment catalog for metadata."""
        self.assessments = self._load_assessment_catalog(assessment_catalog_path)
        
    def _load_assessment_catalog(self, catalog_path: str) -> Dict:
        """Load assessment catalog for explanations."""
        with open(catalog_path, 'r') as f:
            catalog = json.load(f)
            assessments_dict = {}
            for assessment in catalog["assessments"]:
                aid = assessment["assessment_id"]
                assessments_dict[aid] = {
                    "name": assessment["name"],
                    "type": assessment["assessment_type"],
                    "duration": assessment["duration_minutes"],
                    "constructs": assessment["measured_constructs"],
                    "description": assessment["description"],
                    "adverse_impact_risk": assessment["adverse_impact_risk"],
                    "reliability": assessment["reliability"]
                }
            return assessments_dict
    
    def generate_decision_trace(self, 
                              battery_data: Dict,
                              job_skills: Dict[str, float],
                              utility_components: Dict,
                              alternatives: List[Dict] = None) -> Dict:
        """
        Generate comprehensive decision trace for a recommended battery.
        
        Args:
            battery_data: Selected battery with all metadata
            job_skills: Job skill importance weights
            utility_components: Utility function breakdown
            alternatives: Alternative batteries considered
            
        Returns:
            Human-readable decision trace
        """
        battery_id = battery_data["battery_id"]
        assessments = battery_data["assessments"]
        
        trace = {
            "recommendation": {
                "battery_id": battery_id,
                "assessments": assessments,
                "total_duration": battery_data["total_duration"],
                "expected_performance": battery_data["expected_performance"],
                "fairness_risk": battery_data["total_fairness_risk"],
                "utility_score": battery_data.get("total_utility", "N/A")
            },
            "rationale": self._generate_rationale(battery_data, job_skills),
            "skill_coverage": self._explain_skill_coverage(battery_data, job_skills),
            "trade_offs": self._explain_trade_offs(utility_components, battery_data),
            "assessment_details": self._explain_assessments(assessments),
            "decision_factors": self._explain_decision_factors(battery_data, utility_components),
            "alternatives_considered": self._explain_alternatives(alternatives) if alternatives else None,
            "confidence_indicators": self._generate_confidence_indicators(battery_data)
        }
        
        return trace
    
    def _generate_rationale(self, battery_data: Dict, job_skills: Dict[str, float]) -> Dict:
        """Generate high-level rationale for the recommendation."""
        assessments = battery_data["assessments"]
        performance = battery_data["expected_performance"]
        fairness_risk = battery_data["total_fairness_risk"]
        duration = battery_data["total_duration"]
        
        # Determine primary rationale
        if performance > 0.7 and fairness_risk < 0.3:
            primary_reason = "High performance with low fairness risk"
        elif performance > 0.6 and duration < 45:
            primary_reason = "Good performance with time efficiency"
        elif fairness_risk < 0.2:
            primary_reason = "Minimal adverse impact risk"
        elif duration < 30:
            primary_reason = "Time-efficient assessment"
        else:
            primary_reason = "Balanced trade-offs across objectives"
        
        # Assessment composition rationale
        types = [self.assessments[aid]["type"] for aid in assessments]
        type_counts = {t: types.count(t) for t in set(types)}
        
        composition_reason = self._explain_composition(type_counts, len(assessments))
        
        return {
            "primary_reason": primary_reason,
            "composition_rationale": composition_reason,
            "key_strengths": self._identify_key_strengths(battery_data),
            "main_trade_offs": self._identify_main_trade_offs(battery_data)
        }
    
    def _explain_composition(self, type_counts: Dict, total_assessments: int) -> str:
        """Explain why this assessment composition was chosen."""
        if total_assessments == 1:
            return "Single assessment selected for maximum efficiency"
        
        if type_counts.get("cognitive", 0) > 1:
            return "Multiple cognitive assessments for comprehensive ability measurement"
        elif type_counts.get("cognitive", 0) == 1 and type_counts.get("personality", 0) >= 1:
            return "Balanced cognitive-personality combination for holistic evaluation"
        elif type_counts.get("personality", 0) > 1:
            return "Personality-focused battery to minimize adverse impact"
        elif "situational_judgment" in type_counts:
            return "Includes situational judgment for job-relevant behavioral assessment"
        else:
            return "Diverse assessment types for comprehensive evaluation"
    
    def _explain_skill_coverage(self, battery_data: Dict, job_skills: Dict[str, float]) -> Dict:
        """Explain how the battery covers required job skills."""
        skill_coverage = battery_data.get("skill_coverage", {})
        coverage_explanations = {}
        
        # Sort skills by importance
        sorted_skills = sorted(job_skills.items(), key=lambda x: x[1], reverse=True)
        
        for skill_id, importance in sorted_skills:
            coverage = skill_coverage.get(skill_id, 0.0)
            
            # Categorize importance and coverage
            importance_level = "High" if importance > 0.2 else "Moderate" if importance > 0.1 else "Low"
            coverage_level = "Strong" if coverage > 0.7 else "Moderate" if coverage > 0.4 else "Weak"
            
            # Generate explanation
            skill_name = self._get_skill_name(skill_id)
            explanation = f"{importance_level} importance ({importance:.1%}) - {coverage_level.lower()} coverage ({coverage:.1%})"
            
            # Add assessment contribution
            contributing_assessments = self._find_contributing_assessments(skill_id, battery_data["assessments"])
            if contributing_assessments:
                explanation += f" via {', '.join(contributing_assessments)}"
            
            coverage_explanations[skill_name] = explanation
        
        return coverage_explanations
    
    def _explain_trade_offs(self, utility_components: Dict, battery_data: Dict) -> Dict:
        """Explain the key trade-offs made in the recommendation."""
        trade_offs = {}
        
        # Performance vs Fairness
        performance = battery_data["expected_performance"]
        fairness_risk = battery_data["total_fairness_risk"]
        
        if performance > 0.65 and fairness_risk > 0.5:
            trade_offs["performance_vs_fairness"] = "Prioritized performance over fairness - higher adverse impact risk accepted"
        elif performance < 0.6 and fairness_risk < 0.3:
            trade_offs["performance_vs_fairness"] = "Prioritized fairness over performance - some validity sacrificed"
        else:
            trade_offs["performance_vs_fairness"] = "Balanced performance and fairness considerations"
        
        # Time vs Accuracy
        duration = battery_data["total_duration"]
        if duration < 30:
            trade_offs["time_vs_accuracy"] = "Prioritized time efficiency - shorter assessments selected"
        elif duration > 60:
            trade_offs["time_vs_accuracy"] = "Prioritized comprehensive assessment - longer duration accepted"
        else:
            trade_offs["time_vs_accuracy"] = "Balanced time and assessment comprehensiveness"
        
        # Specific decisions
        assessments = battery_data["assessments"]
        cognitive_count = sum(1 for aid in assessments if self.assessments[aid]["type"] == "cognitive")
        
        if cognitive_count == 0:
            trade_offs["assessment_selection"] = "Avoided cognitive tests to minimize adverse impact"
        elif cognitive_count > 2:
            trade_offs["assessment_selection"] = "Multiple cognitive tests for maximum predictive validity"
        else:
            trade_offs["assessment_selection"] = "Limited cognitive testing to balance validity and fairness"
        
        return trade_offs
    
    def _explain_assessments(self, assessments: List[str]) -> Dict:
        """Provide detailed explanation for each selected assessment."""
        explanations = {}
        
        for aid in assessments:
            if aid in self.assessments:
                assessment = self.assessments[aid]
                explanations[aid] = {
                    "name": assessment["name"],
                    "purpose": self._get_assessment_purpose(assessment),
                    "duration": f"{assessment['duration']} minutes",
                    "key_constructs": list(assessment["constructs"].keys())[:3],
                    "selection_reason": self._get_selection_reason(assessment)
                }
        
        return explanations
    
    def _get_assessment_purpose(self, assessment: Dict) -> str:
        """Get human-readable purpose for an assessment."""
        assessment_type = assessment["type"]
        
        if assessment_type == "cognitive":
            return "Measures cognitive abilities and reasoning skills"
        elif assessment_type == "personality":
            return "Evaluates personality traits and work style preferences"
        elif assessment_type == "situational_judgment":
            return "Assesses judgment and decision-making in work scenarios"
        elif assessment_type == "skills_based":
            return "Tests specific job-relevant skills and competencies"
        else:
            return "Evaluates job-relevant capabilities"
    
    def _get_selection_reason(self, assessment: Dict) -> str:
        """Get reason why this specific assessment was selected."""
        reliability = assessment["reliability"]
        risk = assessment["adverse_impact_risk"]
        duration = assessment["duration"]
        
        reasons = []
        
        if reliability > 0.8:
            reasons.append("high reliability")
        if risk < 0.2:
            reasons.append("low adverse impact")
        if duration < 15:
            reasons.append("time efficient")
        
        if not reasons:
            reasons.append("balanced characteristics")
        
        return f"Selected for {' and '.join(reasons)}"
    
    def _explain_decision_factors(self, battery_data: Dict, utility_components: Dict) -> Dict:
        """Explain the key factors that influenced the decision."""
        factors = {}
        
        # Utility component analysis
        components = utility_components.get("components", {})
        
        strongest_factor = max(components.items(), key=lambda x: abs(x[1])) if components else ("performance", 0)
        factors["primary_driver"] = f"{strongest_factor[0].title()} was the primary decision driver"
        
        # Constraint analysis
        duration = battery_data["total_duration"]
        if duration > 75:
            factors["time_constraint"] = "Time constraint was a limiting factor"
        
        # Fairness consideration
        fairness_risk = battery_data["total_fairness_risk"]
        if fairness_risk > 0.6:
            factors["fairness_concern"] = "High fairness risk required careful consideration"
        elif fairness_risk < 0.2:
            factors["fairness_advantage"] = "Low adverse impact risk was a key advantage"
        
        return factors
    
    def _explain_alternatives(self, alternatives: List[Dict]) -> Dict:
        """Explain why alternatives were not selected."""
        if not alternatives:
            return {}
        
        explanations = {}
        
        for alt in alternatives[:3]:  # Top 3 alternatives
            battery_id = alt["battery_id"]
            
            # Compare with selected battery
            reasons = []
            
            if alt["total_fairness_risk"] > 0.6:
                reasons.append("higher fairness risk")
            if alt["total_duration"] > 90:
                reasons.append("excessive duration")
            if alt["expected_performance"] < 0.5:
                reasons.append("lower expected performance")
            
            if not reasons:
                reasons.append("lower overall utility")
            
            explanations[battery_id] = f"Not selected due to {' and '.join(reasons)}"
        
        return explanations
    
    def _generate_confidence_indicators(self, battery_data: Dict) -> Dict:
        """Generate confidence indicators for the recommendation."""
        performance_std = battery_data.get("performance_std", 0.1)
        coverage_score = battery_data.get("coverage_score", 0.5)
        
        confidence = {}
        
        # Performance confidence
        if performance_std < 0.08:
            confidence["performance_certainty"] = "High confidence in performance estimate"
        elif performance_std < 0.12:
            confidence["performance_certainty"] = "Moderate confidence in performance estimate"
        else:
            confidence["performance_certainty"] = "Lower confidence - consider additional assessments"
        
        # Coverage confidence
        if coverage_score > 0.7:
            confidence["skill_coverage"] = "Comprehensive skill coverage achieved"
        elif coverage_score > 0.5:
            confidence["skill_coverage"] = "Adequate skill coverage"
        else:
            confidence["skill_coverage"] = "Limited skill coverage - gaps may exist"
        
        return confidence
    
    def _get_skill_name(self, skill_id: str) -> str:
        """Convert skill ID to human-readable name."""
        skill_names = {
            "C1": "Fluid Intelligence", "C2": "Processing Speed", "C3": "Verbal Ability",
            "C4": "Spatial Ability", "C5": "Memory", "B1": "Conscientiousness",
            "B2": "Emotional Stability", "B3": "Agreeableness", "B4": "Teamwork",
            "B5": "Openness", "W1": "Achievement Focus", "W2": "Leadership",
            "W3": "Adaptability", "W4": "Communication", "W5": "Initiative"
        }
        return skill_names.get(skill_id, skill_id)
    
    def _find_contributing_assessments(self, skill_id: str, assessments: List[str]) -> List[str]:
        """Find which assessments contribute to a specific skill."""
        contributing = []
        for aid in assessments:
            if aid in self.assessments:
                constructs = self.assessments[aid]["constructs"]
                if skill_id in constructs and constructs[skill_id] > 0.3:
                    contributing.append(self.assessments[aid]["name"])
        return contributing
    
    def _identify_key_strengths(self, battery_data: Dict) -> List[str]:
        """Identify key strengths of the recommended battery."""
        strengths = []
        
        if battery_data["expected_performance"] > 0.7:
            strengths.append("High predictive validity")
        if battery_data["total_fairness_risk"] < 0.3:
            strengths.append("Low adverse impact risk")
        if battery_data["total_duration"] < 45:
            strengths.append("Time efficient")
        if battery_data.get("coverage_score", 0) > 0.6:
            strengths.append("Comprehensive skill coverage")
        
        return strengths
    
    def _identify_main_trade_offs(self, battery_data: Dict) -> List[str]:
        """Identify main trade-offs in the recommendation."""
        trade_offs = []
        
        if battery_data["expected_performance"] < 0.6:
            trade_offs.append("Some performance sacrificed for other objectives")
        if battery_data["total_fairness_risk"] > 0.5:
            trade_offs.append("Higher adverse impact risk for better validity")
        if battery_data["total_duration"] > 60:
            trade_offs.append("Longer duration for comprehensive assessment")
        
        return trade_offs
    
    def generate_summary_explanation(self, decision_trace: Dict) -> str:
        """Generate a concise summary explanation for non-technical stakeholders."""
        recommendation = decision_trace["recommendation"]
        rationale = decision_trace["rationale"]
        
        battery_name = recommendation["battery_id"]
        duration = recommendation["total_duration"]
        performance = recommendation["expected_performance"]
        fairness_risk = recommendation["fairness_risk"]
        
        summary = f"""
RECOMMENDATION SUMMARY

Battery: {battery_name}
Duration: {duration} minutes
Expected Performance: {performance:.1%}
Fairness Risk: {fairness_risk:.1%}

WHY THIS BATTERY:
{rationale['primary_reason']}

KEY STRENGTHS:
{chr(10).join(f'• {strength}' for strength in rationale['key_strengths'])}

MAIN TRADE-OFFS:
{chr(10).join(f'• {trade_off}' for trade_off in rationale['main_trade_offs'])}

This recommendation balances job performance prediction with fairness considerations and operational constraints.
        """.strip()
        
        return summary


if __name__ == "__main__":
    # Test decision trace generation
    tracer = DecisionTracer("../data/processed/assessment_catalog.json")
    
    # Sample battery data
    sample_battery = {
        "battery_id": "SHL_NUM_01_SHL_OPQ_01",
        "assessments": ["SHL_NUM_01", "SHL_OPQ_01"],
        "total_duration": 32,
        "expected_performance": 0.68,
        "total_fairness_risk": 0.42,
        "total_utility": 0.45,
        "skill_coverage": {"C1": 0.8, "B1": 0.7, "W1": 0.6},
        "coverage_score": 0.65,
        "performance_std": 0.09
    }
    
    job_skills = {"C1": 0.3, "B1": 0.2, "W1": 0.15}
    utility_components = {
        "components": {"performance": 0.68, "fairness": -0.21, "time": -0.02}
    }
    
    trace = tracer.generate_decision_trace(sample_battery, job_skills, utility_components)
    
    print("DECISION TRACE GENERATED:")
    print(json.dumps(trace, indent=2))
    
    print("\nSUMMARY EXPLANATION:")
    summary = tracer.generate_summary_explanation(trace)
    print(summary)