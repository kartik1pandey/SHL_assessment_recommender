"""
Phase 4.5: Utility Scoring & Selection
Multi-objective utility function for assessment battery optimization.
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class UtilityWeights:
    """Utility function weights for multi-objective optimization."""
    alpha: float = 1.0    # Performance weight
    beta: float = 0.5     # Fairness penalty weight  
    gamma: float = 0.3    # Time cost weight
    delta: float = 0.2    # Constraint penalty weight

class UtilityOptimizer:
    """Multi-objective utility optimizer for assessment batteries."""
    
    def __init__(self, weights: Optional[UtilityWeights] = None):
        """
        Initialize utility optimizer.
        
        Args:
            weights: Utility function weights
        """
        self.weights = weights or UtilityWeights()
        
    def compute_utility(self, 
                       expected_performance: float,
                       fairness_risk: float,
                       time_cost: float,
                       constraint_penalty: float = 0.0) -> Dict[str, float]:
        """
        Compute multi-objective utility for a battery.
        
        U(B) = α·E[P|B] - β·FairnessRisk(B) - γ·TimeCost(B) - δ·ConstraintPenalty(B)
        
        Args:
            expected_performance: Expected job performance [0,1]
            fairness_risk: Fairness risk score [0,1]
            time_cost: Normalized time cost [0,1]
            constraint_penalty: Soft constraint penalty [0,∞)
            
        Returns:
            Utility components and total utility
        """
        # Performance benefit (higher is better)
        performance_utility = self.weights.alpha * expected_performance
        
        # Fairness penalty (higher risk = lower utility)
        fairness_penalty = self.weights.beta * fairness_risk
        
        # Time cost penalty (longer = lower utility)
        time_penalty = self.weights.gamma * time_cost
        
        # Constraint penalty
        constraint_penalty_weighted = self.weights.delta * constraint_penalty
        
        # Total utility
        total_utility = (performance_utility - 
                        fairness_penalty - 
                        time_penalty - 
                        constraint_penalty_weighted)
        
        return {
            "total_utility": total_utility,
            "performance_utility": performance_utility,
            "fairness_penalty": fairness_penalty,
            "time_penalty": time_penalty,
            "constraint_penalty": constraint_penalty_weighted,
            "components": {
                "performance": performance_utility,
                "fairness": -fairness_penalty,
                "time": -time_penalty,
                "constraints": -constraint_penalty_weighted
            }
        }
    
    def normalize_time_cost(self, duration: float, max_duration: float = 120) -> float:
        """Normalize time cost to [0,1] scale."""
        return min(duration / max_duration, 1.0)
    
    def evaluate_battery(self, battery_data: Dict) -> Dict:
        """
        Evaluate a single battery with full utility analysis.
        
        Args:
            battery_data: Battery with performance, fairness, and constraint data
            
        Returns:
            Complete utility evaluation
        """
        # Extract components
        expected_performance = battery_data.get("expected_performance", 0.0)
        fairness_risk = battery_data.get("total_fairness_risk", 0.0)
        duration = battery_data.get("total_duration", 0)
        constraint_penalty = battery_data.get("total_penalty", 0.0)
        
        # Normalize time cost
        time_cost = self.normalize_time_cost(duration)
        
        # Compute utility
        utility_analysis = self.compute_utility(
            expected_performance=expected_performance,
            fairness_risk=fairness_risk,
            time_cost=time_cost,
            constraint_penalty=constraint_penalty
        )
        
        # Add metadata
        utility_analysis.update({
            "battery_id": battery_data.get("battery_id", "unknown"),
            "assessments": battery_data.get("assessments", []),
            "raw_metrics": {
                "expected_performance": expected_performance,
                "fairness_risk": fairness_risk,
                "duration": duration,
                "constraint_penalty": constraint_penalty
            }
        })
        
        return utility_analysis
    
    def rank_batteries(self, batteries_data: List[Dict]) -> List[Dict]:
        """
        Rank batteries by utility score.
        
        Returns:
            Batteries sorted by utility (descending)
        """
        evaluated_batteries = []
        
        for battery_data in batteries_data:
            evaluation = self.evaluate_battery(battery_data)
            
            # Combine original data with utility evaluation
            combined = {**battery_data, **evaluation}
            evaluated_batteries.append(combined)
        
        # Sort by total utility (descending)
        evaluated_batteries.sort(key=lambda x: x["total_utility"], reverse=True)
        
        return evaluated_batteries
    
    def select_top_k(self, batteries_data: List[Dict], k: int = 5) -> List[Dict]:
        """Select top-k batteries by utility."""
        ranked = self.rank_batteries(batteries_data)
        return ranked[:k]
    
    def find_pareto_frontier(self, batteries_data: List[Dict]) -> List[Dict]:
        """
        Find Pareto-efficient batteries.
        
        A battery is Pareto-efficient if no other battery dominates it
        on all objectives (performance, fairness, time).
        
        Returns:
            List of Pareto-efficient batteries
        """
        pareto_batteries = []
        
        for i, battery_i in enumerate(batteries_data):
            is_dominated = False
            
            perf_i = battery_i.get("expected_performance", 0)
            fair_i = 1 - battery_i.get("total_fairness_risk", 1)  # Higher is better
            time_i = 1 - self.normalize_time_cost(battery_i.get("total_duration", 120))  # Higher is better
            
            for j, battery_j in enumerate(batteries_data):
                if i == j:
                    continue
                
                perf_j = battery_j.get("expected_performance", 0)
                fair_j = 1 - battery_j.get("total_fairness_risk", 1)
                time_j = 1 - self.normalize_time_cost(battery_j.get("total_duration", 120))
                
                # Check if j dominates i (j is better or equal on all objectives, strictly better on at least one)
                if (perf_j >= perf_i and fair_j >= fair_i and time_j >= time_i and
                    (perf_j > perf_i or fair_j > fair_i or time_j > time_i)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_batteries.append(battery_i)
        
        return pareto_batteries
    
    def sensitivity_analysis(self, batteries_data: List[Dict], 
                           weight_ranges: Dict[str, Tuple[float, float]]) -> Dict:
        """
        Perform sensitivity analysis on utility weights.
        
        Args:
            batteries_data: List of battery data
            weight_ranges: Dict of weight_name -> (min, max) ranges
            
        Returns:
            Sensitivity analysis results
        """
        results = {
            "weight_scenarios": [],
            "top_battery_changes": [],
            "stability_metrics": {}
        }
        
        # Generate weight scenarios
        scenarios = self._generate_weight_scenarios(weight_ranges)
        
        for scenario_name, weights in scenarios.items():
            # Create optimizer with new weights
            temp_optimizer = UtilityOptimizer(weights)
            
            # Rank batteries with new weights
            ranked = temp_optimizer.rank_batteries(batteries_data)
            top_battery = ranked[0]["battery_id"] if ranked else None
            
            results["weight_scenarios"].append({
                "scenario": scenario_name,
                "weights": weights.__dict__,
                "top_battery": top_battery,
                "top_utility": ranked[0]["total_utility"] if ranked else 0
            })
        
        # Analyze stability
        top_batteries = [s["top_battery"] for s in results["weight_scenarios"]]
        unique_tops = set(top_batteries)
        
        results["stability_metrics"] = {
            "unique_top_batteries": len(unique_tops),
            "most_frequent_top": max(set(top_batteries), key=top_batteries.count),
            "stability_ratio": top_batteries.count(results["stability_metrics"].get("most_frequent_top", "")) / len(top_batteries)
        }
        
        return results
    
    def _generate_weight_scenarios(self, weight_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, UtilityWeights]:
        """Generate weight scenarios for sensitivity analysis."""
        scenarios = {}
        
        # Default scenario
        scenarios["default"] = self.weights
        
        # Performance-focused
        scenarios["performance_focused"] = UtilityWeights(alpha=1.5, beta=0.3, gamma=0.2, delta=0.1)
        
        # Fairness-focused
        scenarios["fairness_focused"] = UtilityWeights(alpha=0.8, beta=1.0, gamma=0.3, delta=0.2)
        
        # Efficiency-focused
        scenarios["efficiency_focused"] = UtilityWeights(alpha=1.0, beta=0.4, gamma=0.8, delta=0.3)
        
        # Balanced
        scenarios["balanced"] = UtilityWeights(alpha=1.0, beta=0.5, gamma=0.5, delta=0.25)
        
        return scenarios
    
    def explain_recommendation(self, battery_data: Dict) -> Dict[str, str]:
        """
        Generate human-readable explanation for battery recommendation.
        
        Returns:
            Dict of explanation components
        """
        evaluation = self.evaluate_battery(battery_data)
        
        explanations = {
            "overall": self._explain_overall_utility(evaluation),
            "performance": self._explain_performance(battery_data),
            "fairness": self._explain_fairness(battery_data),
            "efficiency": self._explain_efficiency(battery_data),
            "trade_offs": self._explain_trade_offs(evaluation)
        }
        
        return explanations
    
    def _explain_overall_utility(self, evaluation: Dict) -> str:
        """Explain overall utility score."""
        utility = evaluation["total_utility"]
        
        if utility > 0.7:
            return f"Excellent battery (utility: {utility:.2f}) - strong across all objectives"
        elif utility > 0.5:
            return f"Good battery (utility: {utility:.2f}) - balanced trade-offs"
        elif utility > 0.3:
            return f"Acceptable battery (utility: {utility:.2f}) - some compromises"
        else:
            return f"Poor battery (utility: {utility:.2f}) - significant limitations"
    
    def _explain_performance(self, battery_data: Dict) -> str:
        """Explain performance characteristics."""
        perf = battery_data.get("expected_performance", 0)
        coverage = battery_data.get("coverage_score", 0)
        
        return f"Expected performance: {perf:.1%} (skill coverage: {coverage:.1%})"
    
    def _explain_fairness(self, battery_data: Dict) -> str:
        """Explain fairness characteristics."""
        risk = battery_data.get("total_fairness_risk", 0)
        
        if risk < 0.3:
            return f"Low fairness risk ({risk:.1%}) - minimal adverse impact concerns"
        elif risk < 0.6:
            return f"Moderate fairness risk ({risk:.1%}) - some adverse impact potential"
        else:
            return f"High fairness risk ({risk:.1%}) - significant adverse impact concerns"
    
    def _explain_efficiency(self, battery_data: Dict) -> str:
        """Explain time efficiency."""
        duration = battery_data.get("total_duration", 0)
        size = battery_data.get("size", 1)
        efficiency = duration / size
        
        return f"Duration: {duration} min ({efficiency:.0f} min/assessment)"
    
    def _explain_trade_offs(self, evaluation: Dict) -> str:
        """Explain key trade-offs."""
        components = evaluation["components"]
        
        strongest = max(components.items(), key=lambda x: abs(x[1]))
        weakest = min(components.items(), key=lambda x: x[1])
        
        return f"Strongest: {strongest[0]} ({strongest[1]:+.2f}), Weakest: {weakest[0]} ({weakest[1]:+.2f})"


if __name__ == "__main__":
    # Test utility optimization
    optimizer = UtilityOptimizer()
    
    # Test battery data
    test_batteries = [
        {
            "battery_id": "high_perf_high_risk",
            "assessments": ["SHL_NUM_01", "SHL_VER_01"],
            "expected_performance": 0.85,
            "total_fairness_risk": 0.7,
            "total_duration": 60,
            "total_penalty": 0.1
        },
        {
            "battery_id": "balanced",
            "assessments": ["SHL_NUM_01", "OPQ_BHV"],
            "expected_performance": 0.75,
            "total_fairness_risk": 0.4,
            "total_duration": 45,
            "total_penalty": 0.0
        },
        {
            "battery_id": "low_risk_low_perf",
            "assessments": ["SJT_GEN", "OPQ_BHV"],
            "expected_performance": 0.65,
            "total_fairness_risk": 0.2,
            "total_duration": 50,
            "total_penalty": 0.0
        }
    ]
    
    # Rank batteries
    ranked = optimizer.rank_batteries(test_batteries)
    
    print("Battery Rankings:")
    for i, battery in enumerate(ranked):
        print(f"{i+1}. {battery['battery_id']}")
        print(f"   Utility: {battery['total_utility']:.3f}")
        print(f"   Performance: {battery['raw_metrics']['expected_performance']:.1%}")
        print(f"   Fairness Risk: {battery['raw_metrics']['fairness_risk']:.1%}")
        print(f"   Duration: {battery['raw_metrics']['duration']} min")
        
        # Get explanation
        explanation = optimizer.explain_recommendation(battery)
        print(f"   Explanation: {explanation['overall']}")
        print()
    
    # Find Pareto frontier
    pareto = optimizer.find_pareto_frontier(test_batteries)
    print(f"Pareto-efficient batteries: {len(pareto)}")
    for battery in pareto:
        print(f"  - {battery['battery_id']}")