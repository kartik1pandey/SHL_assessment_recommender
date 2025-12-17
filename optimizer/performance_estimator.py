"""
Phase 4.2: Expected Performance Aggregation
Estimates expected job performance for assessment batteries using Phase 3 causal model.
"""

import math
from typing import Dict, List, Tuple

class PerformanceEstimator:
    """Estimates expected performance for assessment batteries."""
    
    def __init__(self, job_skill_distribution: Dict[str, float]):
        """
        Initialize performance estimator.
        
        Args:
            job_skill_distribution: Job-specific skill importance weights
        """
        self.job_skills = job_skill_distribution
        
        # Diminishing returns parameters
        self.base_performance = 0.5  # Baseline without assessments
        self.max_performance = 0.9   # Theoretical maximum
        self.diminishing_factor = 0.7  # Controls diminishing returns
        
    def estimate_battery_performance(self, battery_metadata: Dict) -> Dict:
        """
        Estimate expected performance for a battery.
        
        Args:
            battery_metadata: Battery metadata from BatteryGenerator
            
        Returns:
            Performance estimates with uncertainty
        """
        assessments = battery_metadata["assessments"]
        
        # Get skill coverage from battery
        skill_coverage = self._compute_skill_coverage(assessments)
        
        # Estimate performance with diminishing returns
        expected_performance = self._compute_expected_performance(skill_coverage)
        
        # Estimate uncertainty (decreases with more assessments)
        performance_uncertainty = self._compute_uncertainty(len(assessments))
        
        return {
            "expected_performance": expected_performance,
            "performance_std": performance_uncertainty,
            "skill_coverage": skill_coverage,
            "coverage_score": sum(skill_coverage.values()) / len(skill_coverage)
        }
    
    def _compute_skill_coverage(self, assessments: List[str]) -> Dict[str, float]:
        """
        Compute how well battery covers required job skills.
        
        Returns:
            Dict mapping skill_id -> coverage_score [0,1]
        """
        skill_coverage = {}
        
        for skill_id in self.job_skills:
            coverage = 0.0
            
            # Each assessment contributes to skill coverage
            for assessment_id in assessments:
                # Get assessment-skill relationship strength
                relationship_strength = self._get_assessment_skill_strength(
                    assessment_id, skill_id
                )
                coverage += relationship_strength
            
            # Apply diminishing returns: additional assessments help less
            skill_coverage[skill_id] = 1 - math.exp(-self.diminishing_factor * coverage)
        
        return skill_coverage
    
    def _get_assessment_skill_strength(self, assessment_id: str, skill_id: str) -> float:
        """
        Get strength of assessment-skill relationship.
        
        This would ideally come from Phase 3 causal model weights.
        For now, use heuristic based on assessment type and skill type.
        """
        # Heuristic mapping (in real implementation, use causal model)
        assessment_skill_map = {
            # Cognitive assessments
            ("SHL_NUM_01", "C1"): 0.8, ("SHL_NUM_01", "C2"): 0.6,
            ("SHL_VER_01", "C1"): 0.7, ("SHL_VER_01", "C3"): 0.8,
            ("SHL_LOG_01", "C2"): 0.9, ("SHL_LOG_01", "C4"): 0.7,
            
            # Personality assessments
            ("OPQ_BHV", "B1"): 0.8, ("OPQ_BHV", "B2"): 0.7,
            ("NEO_FFI", "B3"): 0.8, ("NEO_FFI", "W1"): 0.6,
            
            # Situational assessments
            ("SJT_GEN", "B4"): 0.7, ("SJT_GEN", "W2"): 0.8,
            ("SJT_LEAD", "B5"): 0.9, ("SJT_LEAD", "W3"): 0.8,
        }
        
        return assessment_skill_map.get((assessment_id, skill_id), 0.1)
    
    def _compute_expected_performance(self, skill_coverage: Dict[str, float]) -> float:
        """
        Compute expected job performance from skill coverage.
        
        Uses job skill importance weights and diminishing returns.
        """
        weighted_coverage = 0.0
        total_weight = 0.0
        
        for skill_id, importance in self.job_skills.items():
            coverage = skill_coverage.get(skill_id, 0.0)
            weighted_coverage += importance * coverage
            total_weight += importance
        
        # Normalize by total importance
        if total_weight > 0:
            normalized_coverage = weighted_coverage / total_weight
        else:
            normalized_coverage = 0.0
        
        # Apply performance scaling with diminishing returns
        performance_gain = (self.max_performance - self.base_performance) * normalized_coverage
        expected_performance = self.base_performance + performance_gain
        
        return min(expected_performance, self.max_performance)
    
    def _compute_uncertainty(self, num_assessments: int) -> float:
        """
        Compute performance prediction uncertainty.
        
        Uncertainty decreases with more assessments (but with diminishing returns).
        """
        base_uncertainty = 0.15  # High uncertainty with no assessments
        min_uncertainty = 0.05   # Minimum uncertainty with many assessments
        
        # Exponential decay in uncertainty
        uncertainty = min_uncertainty + (base_uncertainty - min_uncertainty) * math.exp(-0.5 * num_assessments)
        
        return uncertainty
    
    def compare_batteries(self, batteries_metadata: List[Dict]) -> List[Dict]:
        """
        Compare multiple batteries on performance metrics.
        
        Returns:
            Sorted list of batteries with performance estimates
        """
        results = []
        
        for battery_meta in batteries_metadata:
            perf_est = self.estimate_battery_performance(battery_meta)
            
            result = {
                **battery_meta,
                **perf_est,
                "efficiency": perf_est["expected_performance"] / battery_meta["total_duration"]
            }
            results.append(result)
        
        # Sort by expected performance (descending)
        results.sort(key=lambda x: x["expected_performance"], reverse=True)
        
        return results


if __name__ == "__main__":
    # Test performance estimation
    job_skills = {
        "C1": 0.3, "C2": 0.2, "B1": 0.2, "B2": 0.15, "W1": 0.15
    }
    
    estimator = PerformanceEstimator(job_skills)
    
    # Test single battery
    test_battery = {
        "battery_id": "SHL_NUM_01_OPQ_BHV",
        "assessments": ["SHL_NUM_01", "OPQ_BHV"],
        "size": 2,
        "total_duration": 45
    }
    
    performance = estimator.estimate_battery_performance(test_battery)
    print("Performance Estimate:")
    print(f"  Expected: {performance['expected_performance']:.3f}")
    print(f"  Uncertainty: {performance['performance_std']:.3f}")
    print(f"  Coverage: {performance['coverage_score']:.3f}")