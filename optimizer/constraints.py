"""
Phase 4.4: Time & Operational Constraints
Enforces hard and soft constraints on assessment batteries.
"""

from typing import Dict, List, Tuple, Optional
import math

class ConstraintManager:
    """Manages operational constraints for assessment batteries."""
    
    def __init__(self, 
                 max_duration: int = 90,
                 ideal_duration: int = 60,
                 max_assessments: int = 4,
                 min_assessments: int = 1):
        """
        Initialize constraint manager.
        
        Args:
            max_duration: Hard cap on total duration (minutes)
            ideal_duration: Ideal duration for soft penalty (minutes)
            max_assessments: Maximum number of assessments
            min_assessments: Minimum number of assessments
        """
        self.max_duration = max_duration
        self.ideal_duration = ideal_duration
        self.max_assessments = max_assessments
        self.min_assessments = min_assessments
        
        # Penalty parameters
        self.duration_penalty_rate = 0.01  # Penalty per minute over ideal
        self.size_penalty_rate = 0.05      # Penalty per assessment over ideal
        
    def check_hard_constraints(self, battery_metadata: Dict) -> Dict[str, bool]:
        """
        Check if battery violates hard constraints.
        
        Returns:
            Dict of constraint_name -> is_satisfied
        """
        duration = battery_metadata["total_duration"]
        size = battery_metadata["size"]
        
        constraints = {
            "max_duration": duration <= self.max_duration,
            "max_assessments": size <= self.max_assessments,
            "min_assessments": size >= self.min_assessments,
            "has_diversity": self._check_diversity_constraint(battery_metadata)
        }
        
        return constraints
    
    def _check_diversity_constraint(self, battery_metadata: Dict) -> bool:
        """
        Check diversity constraint: no single-type batteries over size 1.
        
        Prevents batteries like [cognitive, cognitive, cognitive].
        """
        if battery_metadata["size"] <= 1:
            return True
        
        types = battery_metadata.get("types", [])
        unique_types = set(types)
        
        # Must have at least 2 different types if size > 2
        if battery_metadata["size"] > 2:
            return len(unique_types) >= 2
        
        return True
    
    def is_feasible(self, battery_metadata: Dict) -> bool:
        """Check if battery satisfies all hard constraints."""
        constraints = self.check_hard_constraints(battery_metadata)
        return all(constraints.values())
    
    def compute_soft_penalties(self, battery_metadata: Dict) -> Dict[str, float]:
        """
        Compute soft constraint penalties.
        
        Returns:
            Dict of penalty_type -> penalty_value
        """
        duration = battery_metadata["total_duration"]
        size = battery_metadata["size"]
        
        penalties = {}
        
        # Duration penalty (linear beyond ideal)
        if duration > self.ideal_duration:
            excess_duration = duration - self.ideal_duration
            penalties["duration"] = self.duration_penalty_rate * excess_duration
        else:
            penalties["duration"] = 0.0
        
        # Size penalty (prefer smaller batteries)
        ideal_size = 2  # Prefer 2-assessment batteries
        if size > ideal_size:
            excess_size = size - ideal_size
            penalties["size"] = self.size_penalty_rate * excess_size
        else:
            penalties["size"] = 0.0
        
        # Efficiency penalty (duration per assessment)
        efficiency = duration / size if size > 0 else float('inf')
        if efficiency > 30:  # More than 30 min per assessment
            penalties["efficiency"] = 0.02 * (efficiency - 30)
        else:
            penalties["efficiency"] = 0.0
        
        return penalties
    
    def get_total_penalty(self, battery_metadata: Dict) -> float:
        """Get total soft constraint penalty."""
        penalties = self.compute_soft_penalties(battery_metadata)
        return sum(penalties.values())
    
    def filter_feasible_batteries(self, batteries_metadata: List[Dict]) -> List[Dict]:
        """
        Filter batteries to only feasible ones.
        
        Returns:
            List of batteries satisfying hard constraints
        """
        feasible = []
        
        for battery_meta in batteries_metadata:
            if self.is_feasible(battery_meta):
                # Add constraint analysis
                battery_meta["constraints"] = self.check_hard_constraints(battery_meta)
                battery_meta["soft_penalties"] = self.compute_soft_penalties(battery_meta)
                battery_meta["total_penalty"] = self.get_total_penalty(battery_meta)
                
                feasible.append(battery_meta)
        
        return feasible
    
    def get_constraint_summary(self, battery_metadata: Dict) -> Dict:
        """Get comprehensive constraint analysis for a battery."""
        hard_constraints = self.check_hard_constraints(battery_metadata)
        soft_penalties = self.compute_soft_penalties(battery_metadata)
        
        return {
            "is_feasible": all(hard_constraints.values()),
            "hard_constraints": hard_constraints,
            "soft_penalties": soft_penalties,
            "total_penalty": sum(soft_penalties.values()),
            "constraint_violations": [
                name for name, satisfied in hard_constraints.items() 
                if not satisfied
            ],
            "penalty_breakdown": {
                name: penalty for name, penalty in soft_penalties.items() 
                if penalty > 0
            }
        }
    
    def recommend_constraint_fixes(self, battery_metadata: Dict) -> List[str]:
        """
        Recommend fixes for constraint violations.
        
        Returns:
            List of actionable recommendations
        """
        summary = self.get_constraint_summary(battery_metadata)
        recommendations = []
        
        if not summary["is_feasible"]:
            violations = summary["constraint_violations"]
            
            if "max_duration" in violations:
                recommendations.append(
                    f"Reduce duration: {battery_metadata['total_duration']} > {self.max_duration} min"
                )
            
            if "max_assessments" in violations:
                recommendations.append(
                    f"Reduce assessments: {battery_metadata['size']} > {self.max_assessments}"
                )
            
            if "has_diversity" in violations:
                recommendations.append("Add assessment diversity (mix cognitive/personality/situational)")
        
        # Soft penalty recommendations
        penalties = summary["penalty_breakdown"]
        
        if "duration" in penalties:
            recommendations.append(
                f"Consider shorter assessments (current: {battery_metadata['total_duration']} min)"
            )
        
        if "size" in penalties:
            recommendations.append("Consider reducing battery size for efficiency")
        
        if "efficiency" in penalties:
            avg_duration = battery_metadata['total_duration'] / battery_metadata['size']
            recommendations.append(f"Low efficiency: {avg_duration:.1f} min/assessment")
        
        if not recommendations:
            recommendations.append("Battery meets all constraints")
        
        return recommendations


class AdaptiveConstraints(ConstraintManager):
    """Constraint manager that adapts based on job requirements."""
    
    def __init__(self, job_context: Dict, **kwargs):
        """
        Initialize adaptive constraints based on job context.
        
        Args:
            job_context: Job-specific requirements
        """
        # Adapt constraints based on job level and type
        job_level = job_context.get("level", "mid")
        job_type = job_context.get("type", "general")
        
        # Senior roles can have longer assessments
        if job_level == "senior":
            kwargs.setdefault("max_duration", 120)
            kwargs.setdefault("ideal_duration", 90)
        elif job_level == "entry":
            kwargs.setdefault("max_duration", 60)
            kwargs.setdefault("ideal_duration", 45)
        
        # Technical roles may need more assessments
        if job_type == "technical":
            kwargs.setdefault("max_assessments", 5)
        
        super().__init__(**kwargs)
        self.job_context = job_context
    
    def _check_diversity_constraint(self, battery_metadata: Dict) -> bool:
        """Job-specific diversity constraints."""
        base_diversity = super()._check_diversity_constraint(battery_metadata)
        
        # Technical jobs require at least one cognitive assessment
        if self.job_context.get("type") == "technical":
            has_cognitive = battery_metadata.get("has_cognitive", False)
            return base_diversity and has_cognitive
        
        # Leadership jobs require personality assessment
        if self.job_context.get("type") == "leadership":
            has_personality = battery_metadata.get("has_personality", False)
            return base_diversity and has_personality
        
        return base_diversity


if __name__ == "__main__":
    # Test constraint management
    constraint_manager = ConstraintManager()
    
    # Test feasible battery
    feasible_battery = {
        "battery_id": "SHL_NUM_01_OPQ_BHV",
        "assessments": ["SHL_NUM_01", "OPQ_BHV"],
        "size": 2,
        "total_duration": 45,
        "types": ["cognitive", "personality"],
        "has_cognitive": True,
        "has_personality": True
    }
    
    # Test infeasible battery (too long)
    infeasible_battery = {
        "battery_id": "LONG_BATTERY",
        "assessments": ["SHL_NUM_01", "SHL_VER_01", "SHL_LOG_01", "OPQ_BHV"],
        "size": 4,
        "total_duration": 120,
        "types": ["cognitive", "cognitive", "cognitive", "personality"],
        "has_cognitive": True,
        "has_personality": True
    }
    
    print("Feasible Battery Analysis:")
    feasible_summary = constraint_manager.get_constraint_summary(feasible_battery)
    print(f"  Is Feasible: {feasible_summary['is_feasible']}")
    print(f"  Total Penalty: {feasible_summary['total_penalty']:.3f}")
    
    print("\nInfeasible Battery Analysis:")
    infeasible_summary = constraint_manager.get_constraint_summary(infeasible_battery)
    print(f"  Is Feasible: {infeasible_summary['is_feasible']}")
    print(f"  Violations: {infeasible_summary['constraint_violations']}")
    
    print("\nRecommendations:")
    recommendations = constraint_manager.recommend_constraint_fixes(infeasible_battery)
    for rec in recommendations:
        print(f"  - {rec}")
    
    # Test adaptive constraints
    print("\nAdaptive Constraints (Technical Job):")
    tech_job = {"level": "senior", "type": "technical"}
    adaptive_manager = AdaptiveConstraints(tech_job)
    
    tech_summary = adaptive_manager.get_constraint_summary(feasible_battery)
    print(f"  Max Duration: {adaptive_manager.max_duration} min")
    print(f"  Is Feasible: {tech_summary['is_feasible']}")