"""
Phase 5.2: Counterfactual Explanations
Answers "What would change the recommendation?" by showing how parameter changes affect battery selection.
"""

from typing import Dict, List, Tuple, Any
import sys
from pathlib import Path

# Add optimizer modules
sys.path.append(str(Path(__file__).parent.parent / "optimizer"))
from utility import UtilityOptimizer, UtilityWeights

class CounterfactualExplainer:
    """Generates counterfactual explanations for assessment battery recommendations."""
    
    def __init__(self, complete_evaluations: List[Dict]):
        """
        Initialize with evaluated battery candidates.
        
        Args:
            complete_evaluations: List of batteries with all metrics computed
        """
        self.batteries = complete_evaluations
        self.base_optimizer = UtilityOptimizer()
        
    def generate_counterfactuals(self, 
                                base_weights: UtilityWeights,
                                parameter_variations: Dict[str, List[float]]) -> Dict:
        """
        Generate counterfactual explanations by varying utility parameters.
        
        Args:
            base_weights: Baseline utility weights
            parameter_variations: Dict of parameter -> list of values to test
            
        Returns:
            Counterfactual analysis results
        """
        base_optimizer = UtilityOptimizer(base_weights)
        base_recommendation = base_optimizer.rank_batteries(self.batteries)[0]
        
        counterfactuals = {
            "base_recommendation": {
                "battery_id": base_recommendation["battery_id"],
                "weights": base_weights.__dict__,
                "utility": base_recommendation["total_utility"],
                "performance": base_recommendation["expected_performance"],
                "fairness_risk": base_recommendation["total_fairness_risk"],
                "duration": base_recommendation["total_duration"]
            },
            "parameter_effects": {},
            "decision_boundaries": {},
            "sensitivity_analysis": {}
        }
        
        # Test each parameter variation
        for param_name, values in parameter_variations.items():
            counterfactuals["parameter_effects"][param_name] = self._analyze_parameter_effect(
                base_weights, param_name, values, base_recommendation
            )
        
        # Find decision boundaries
        counterfactuals["decision_boundaries"] = self._find_decision_boundaries(
            base_weights, base_recommendation
        )
        
        # Overall sensitivity
        counterfactuals["sensitivity_analysis"] = self._compute_sensitivity_metrics(
            base_weights, parameter_variations
        )
        
        return counterfactuals
    
    def _analyze_parameter_effect(self, 
                                 base_weights: UtilityWeights,
                                 param_name: str,
                                 values: List[float],
                                 base_recommendation: Dict) -> Dict:
        """Analyze effect of varying a single parameter."""
        effects = {
            "parameter": param_name,
            "base_value": getattr(base_weights, param_name),
            "variations": [],
            "recommendation_changes": [],
            "stability_score": 0.0
        }
        
        base_battery_id = base_recommendation["battery_id"]
        same_recommendation_count = 0
        
        for value in values:
            # Create modified weights
            modified_weights = UtilityWeights(
                alpha=base_weights.alpha,
                beta=base_weights.beta,
                gamma=base_weights.gamma,
                delta=base_weights.delta
            )
            setattr(modified_weights, param_name, value)
            
            # Get new recommendation
            optimizer = UtilityOptimizer(modified_weights)
            new_recommendation = optimizer.rank_batteries(self.batteries)[0]
            
            # Record change
            changed = new_recommendation["battery_id"] != base_battery_id
            if not changed:
                same_recommendation_count += 1
            
            variation_result = {
                "value": value,
                "new_battery": new_recommendation["battery_id"],
                "changed": changed,
                "utility_change": new_recommendation["total_utility"] - base_recommendation["total_utility"],
                "performance_change": new_recommendation["expected_performance"] - base_recommendation["expected_performance"],
                "fairness_change": new_recommendation["total_fairness_risk"] - base_recommendation["total_fairness_risk"],
                "explanation": self._explain_change(param_name, value, changed, new_recommendation, base_recommendation)
            }
            
            effects["variations"].append(variation_result)
            
            if changed:
                effects["recommendation_changes"].append({
                    "threshold": value,
                    "from": base_battery_id,
                    "to": new_recommendation["battery_id"],
                    "reason": self._explain_why_changed(param_name, value, new_recommendation, base_recommendation)
                })
        
        # Compute stability score
        effects["stability_score"] = same_recommendation_count / len(values)
        
        return effects
    
    def _explain_change(self, param_name: str, value: float, changed: bool, 
                       new_rec: Dict, base_rec: Dict) -> str:
        """Explain what happened when parameter changed."""
        if not changed:
            return f"No change in recommendation despite {param_name} = {value}"
        
        explanations = {
            "alpha": f"Higher performance weight ({value}) favored {new_rec['battery_id']} for better validity",
            "beta": f"Higher fairness penalty ({value}) switched to {new_rec['battery_id']} for lower adverse impact",
            "gamma": f"Higher time penalty ({value}) switched to {new_rec['battery_id']} for shorter duration",
            "delta": f"Higher constraint penalty ({value}) switched to {new_rec['battery_id']} for better compliance"
        }
        
        base_explanation = explanations.get(param_name, f"{param_name} = {value} caused switch to {new_rec['battery_id']}")
        
        # Add specific metrics
        if param_name == "beta":  # Fairness
            fairness_improvement = base_rec["total_fairness_risk"] - new_rec["total_fairness_risk"]
            if fairness_improvement > 0:
                base_explanation += f" (fairness risk reduced by {fairness_improvement:.3f})"
        elif param_name == "gamma":  # Time
            time_improvement = base_rec["total_duration"] - new_rec["total_duration"]
            if time_improvement > 0:
                base_explanation += f" (duration reduced by {time_improvement} minutes)"
        
        return base_explanation
    
    def _explain_why_changed(self, param_name: str, value: float, 
                           new_rec: Dict, base_rec: Dict) -> str:
        """Explain why the recommendation changed at this parameter value."""
        if param_name == "beta":  # Fairness penalty
            if new_rec["total_fairness_risk"] < base_rec["total_fairness_risk"]:
                return f"Fairness penalty {value} made lower-risk battery more attractive"
        elif param_name == "gamma":  # Time penalty
            if new_rec["total_duration"] < base_rec["total_duration"]:
                return f"Time penalty {value} made shorter battery more attractive"
        elif param_name == "alpha":  # Performance weight
            if new_rec["expected_performance"] > base_rec["expected_performance"]:
                return f"Performance weight {value} made higher-validity battery more attractive"
        
        return f"Parameter {param_name} = {value} shifted utility balance"
    
    def _find_decision_boundaries(self, base_weights: UtilityWeights, 
                                base_recommendation: Dict) -> Dict:
        """Find critical parameter values where recommendation changes."""
        boundaries = {}
        
        # Test fairness penalty boundary
        boundaries["fairness_threshold"] = self._find_parameter_boundary(
            base_weights, "beta", base_recommendation, 0.0, 2.0, 0.1
        )
        
        # Test performance weight boundary
        boundaries["performance_threshold"] = self._find_parameter_boundary(
            base_weights, "alpha", base_recommendation, 0.5, 2.0, 0.1
        )
        
        # Test time penalty boundary
        boundaries["time_threshold"] = self._find_parameter_boundary(
            base_weights, "gamma", base_recommendation, 0.0, 1.5, 0.1
        )
        
        return boundaries
    
    def _find_parameter_boundary(self, base_weights: UtilityWeights, param_name: str,
                               base_recommendation: Dict, min_val: float, max_val: float,
                               step: float) -> Dict:
        """Find the boundary value where recommendation changes."""
        base_battery_id = base_recommendation["battery_id"]
        base_value = getattr(base_weights, param_name)
        
        # Search upward from base value
        boundary_up = None
        current_val = base_value + step
        while current_val <= max_val:
            modified_weights = UtilityWeights(
                alpha=base_weights.alpha,
                beta=base_weights.beta,
                gamma=base_weights.gamma,
                delta=base_weights.delta
            )
            setattr(modified_weights, param_name, current_val)
            
            optimizer = UtilityOptimizer(modified_weights)
            new_rec = optimizer.rank_batteries(self.batteries)[0]
            
            if new_rec["battery_id"] != base_battery_id:
                boundary_up = {
                    "value": current_val,
                    "new_battery": new_rec["battery_id"],
                    "direction": "increase"
                }
                break
            
            current_val += step
        
        # Search downward from base value
        boundary_down = None
        current_val = base_value - step
        while current_val >= min_val:
            modified_weights = UtilityWeights(
                alpha=base_weights.alpha,
                beta=base_weights.beta,
                gamma=base_weights.gamma,
                delta=base_weights.delta
            )
            setattr(modified_weights, param_name, current_val)
            
            optimizer = UtilityOptimizer(modified_weights)
            new_rec = optimizer.rank_batteries(self.batteries)[0]
            
            if new_rec["battery_id"] != base_battery_id:
                boundary_down = {
                    "value": current_val,
                    "new_battery": new_rec["battery_id"],
                    "direction": "decrease"
                }
                break
            
            current_val -= step
        
        return {
            "parameter": param_name,
            "base_value": base_value,
            "boundary_up": boundary_up,
            "boundary_down": boundary_down,
            "stability_range": self._compute_stability_range(boundary_up, boundary_down, base_value)
        }
    
    def _compute_stability_range(self, boundary_up: Dict, boundary_down: Dict, base_value: float) -> Dict:
        """Compute the stability range for a parameter."""
        upper_bound = boundary_up["value"] if boundary_up else float('inf')
        lower_bound = boundary_down["value"] if boundary_down else float('-inf')
        
        return {
            "stable_range": (lower_bound, upper_bound),
            "range_width": upper_bound - lower_bound if boundary_up and boundary_down else "unbounded",
            "relative_stability": "high" if (upper_bound - lower_bound) > base_value else "moderate"
        }
    
    def _compute_sensitivity_metrics(self, base_weights: UtilityWeights,
                                   parameter_variations: Dict[str, List[float]]) -> Dict:
        """Compute overall sensitivity metrics."""
        total_variations = sum(len(values) for values in parameter_variations.values())
        total_changes = 0
        
        base_optimizer = UtilityOptimizer(base_weights)
        base_battery = base_optimizer.rank_batteries(self.batteries)[0]["battery_id"]
        
        for param_name, values in parameter_variations.items():
            for value in values:
                modified_weights = UtilityWeights(
                    alpha=base_weights.alpha,
                    beta=base_weights.beta,
                    gamma=base_weights.gamma,
                    delta=base_weights.delta
                )
                setattr(modified_weights, param_name, value)
                
                optimizer = UtilityOptimizer(modified_weights)
                new_battery = optimizer.rank_batteries(self.batteries)[0]["battery_id"]
                
                if new_battery != base_battery:
                    total_changes += 1
        
        stability_score = 1 - (total_changes / total_variations)
        
        return {
            "overall_stability": stability_score,
            "stability_category": "high" if stability_score > 0.7 else "moderate" if stability_score > 0.4 else "low",
            "total_variations_tested": total_variations,
            "recommendation_changes": total_changes,
            "interpretation": self._interpret_sensitivity(stability_score)
        }
    
    def _interpret_sensitivity(self, stability_score: float) -> str:
        """Interpret the sensitivity score."""
        if stability_score > 0.8:
            return "Very stable - recommendation rarely changes with parameter variations"
        elif stability_score > 0.6:
            return "Moderately stable - recommendation changes only with significant parameter shifts"
        elif stability_score > 0.4:
            return "Somewhat sensitive - recommendation changes with moderate parameter variations"
        else:
            return "Highly sensitive - recommendation frequently changes with parameter variations"
    
    def generate_counterfactual_scenarios(self) -> List[Dict]:
        """Generate specific counterfactual scenarios for explanation."""
        scenarios = [
            {
                "scenario": "What if fairness is the top priority?",
                "weights": UtilityWeights(alpha=0.8, beta=1.2, gamma=0.3, delta=0.2),
                "explanation": "Increasing fairness penalty to 1.2 prioritizes low adverse impact"
            },
            {
                "scenario": "What if we have very limited time?",
                "weights": UtilityWeights(alpha=1.0, beta=0.5, gamma=1.0, delta=0.3),
                "explanation": "High time penalty (1.0) strongly favors shorter assessments"
            },
            {
                "scenario": "What if performance is everything?",
                "weights": UtilityWeights(alpha=2.0, beta=0.2, gamma=0.2, delta=0.1),
                "explanation": "Maximum performance weight (2.0) prioritizes validity over all else"
            },
            {
                "scenario": "What if we ignore fairness completely?",
                "weights": UtilityWeights(alpha=1.5, beta=0.0, gamma=0.3, delta=0.2),
                "explanation": "Zero fairness penalty removes adverse impact considerations"
            }
        ]
        
        results = []
        base_optimizer = UtilityOptimizer()
        base_rec = base_optimizer.rank_batteries(self.batteries)[0]
        
        for scenario in scenarios:
            optimizer = UtilityOptimizer(scenario["weights"])
            new_rec = optimizer.rank_batteries(self.batteries)[0]
            
            results.append({
                "scenario": scenario["scenario"],
                "explanation": scenario["explanation"],
                "base_battery": base_rec["battery_id"],
                "new_battery": new_rec["battery_id"],
                "changed": new_rec["battery_id"] != base_rec["battery_id"],
                "performance_change": new_rec["expected_performance"] - base_rec["expected_performance"],
                "fairness_change": new_rec["total_fairness_risk"] - base_rec["total_fairness_risk"],
                "time_change": new_rec["total_duration"] - base_rec["total_duration"],
                "impact_summary": self._summarize_scenario_impact(new_rec, base_rec)
            })
        
        return results
    
    def _summarize_scenario_impact(self, new_rec: Dict, base_rec: Dict) -> str:
        """Summarize the impact of a counterfactual scenario."""
        if new_rec["battery_id"] == base_rec["battery_id"]:
            return "No change in recommendation"
        
        impacts = []
        
        perf_change = new_rec["expected_performance"] - base_rec["expected_performance"]
        if abs(perf_change) > 0.05:
            direction = "increased" if perf_change > 0 else "decreased"
            impacts.append(f"performance {direction} by {abs(perf_change):.3f}")
        
        fair_change = new_rec["total_fairness_risk"] - base_rec["total_fairness_risk"]
        if abs(fair_change) > 0.1:
            direction = "increased" if fair_change > 0 else "reduced"
            impacts.append(f"fairness risk {direction} by {abs(fair_change):.3f}")
        
        time_change = new_rec["total_duration"] - base_rec["total_duration"]
        if abs(time_change) > 5:
            direction = "increased" if time_change > 0 else "reduced"
            impacts.append(f"duration {direction} by {abs(time_change)} minutes")
        
        if impacts:
            return f"Switched to {new_rec['battery_id']}: {', '.join(impacts)}"
        else:
            return f"Switched to {new_rec['battery_id']} with minimal metric changes"


if __name__ == "__main__":
    # Test counterfactual explanations
    print("Testing counterfactual explanations...")
    
    # Mock battery data for testing
    mock_batteries = [
        {
            "battery_id": "high_perf_high_risk",
            "expected_performance": 0.8,
            "total_fairness_risk": 0.7,
            "total_duration": 60,
            "total_penalty": 0.1
        },
        {
            "battery_id": "balanced",
            "expected_performance": 0.7,
            "total_fairness_risk": 0.4,
            "total_duration": 45,
            "total_penalty": 0.0
        },
        {
            "battery_id": "low_risk_fast",
            "expected_performance": 0.6,
            "total_fairness_risk": 0.2,
            "total_duration": 30,
            "total_penalty": 0.0
        }
    ]
    
    explainer = CounterfactualExplainer(mock_batteries)
    
    # Test parameter variations
    base_weights = UtilityWeights(alpha=1.0, beta=0.5, gamma=0.3, delta=0.2)
    variations = {
        "beta": [0.0, 0.3, 0.5, 0.8, 1.0, 1.5],  # Fairness penalty
        "gamma": [0.1, 0.3, 0.5, 0.8, 1.0]       # Time penalty
    }
    
    counterfactuals = explainer.generate_counterfactuals(base_weights, variations)
    
    print("Counterfactual Analysis Results:")
    print(f"Base recommendation: {counterfactuals['base_recommendation']['battery_id']}")
    
    for param, effects in counterfactuals["parameter_effects"].items():
        print(f"\n{param.upper()} Effects:")
        print(f"  Stability score: {effects['stability_score']:.2f}")
        print(f"  Recommendation changes: {len(effects['recommendation_changes'])}")
        
        for change in effects["recommendation_changes"]:
            print(f"    At {param}={change['threshold']}: {change['from']} → {change['to']}")
            print(f"      Reason: {change['reason']}")
    
    # Test scenarios
    print("\nCounterfactual Scenarios:")
    scenarios = explainer.generate_counterfactual_scenarios()
    for scenario in scenarios:
        print(f"\n{scenario['scenario']}")
        print(f"  {scenario['explanation']}")
        if scenario['changed']:
            print(f"  Result: {scenario['base_battery']} → {scenario['new_battery']}")
            print(f"  Impact: {scenario['impact_summary']}")
        else:
            print(f"  Result: No change from {scenario['base_battery']}")