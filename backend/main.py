"""
Assessment Recommendation Engine - FastAPI Backend
RAG-based API for intelligent assessment recommendations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add project modules to path
backend_dir = Path(__file__).parent
project_root = backend_dir.parent
sys.path.insert(0, str(project_root))

# Import core engine components
from optimizer.battery_generator import BatteryGenerator
from optimizer.performance_estimator import PerformanceEstimator
from optimizer.fairness import FairnessAnalyzer
from optimizer.constraints import ConstraintManager
from optimizer.utility import UtilityOptimizer, UtilityWeights
from explainability.decision_trace import DecisionTracer

# Import RAG components
try:
    from rag.rag_engine import RAGEngine
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("Warning: RAG engine not available")

# Initialize FastAPI
app = FastAPI(
    title="Assessment Recommendation Engine API",
    description="Research-grade assessment selection with causal modeling and fairness awareness",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas
class SkillExtractionRequest(BaseModel):
    job_description: str = Field(..., description="Job description text")

class SkillExtractionResponse(BaseModel):
    skills: Dict[str, float] = Field(..., description="Skill distribution")
    entropy: float = Field(..., description="Distribution entropy")
    top_skills: List[Dict[str, float]] = Field(..., description="Top 5 skills")

class PerformanceSimulationRequest(BaseModel):
    skills: Dict[str, float] = Field(..., description="Skill distribution")
    battery_id: str = Field(..., description="Battery identifier")

class PerformanceSimulationResponse(BaseModel):
    expected_performance: float
    uncertainty: float
    skill_coverage: Dict[str, float]

class OptimizationRequest(BaseModel):
    skills: Dict[str, float] = Field(..., description="Job skill requirements")
    fairness_weight: float = Field(0.5, ge=0.0, le=2.0)
    time_weight: float = Field(0.3, ge=0.0, le=1.0)
    max_duration: int = Field(90, ge=30, le=120)

class BatteryRecommendation(BaseModel):
    battery_id: str
    assessments: List[str]
    expected_performance: float
    fairness_risk: float
    duration: int
    utility: float

class OptimizationResponse(BaseModel):
    top_recommendation: BatteryRecommendation
    alternatives: List[BatteryRecommendation]
    pareto_frontier: List[Dict[str, float]]
    trade_off_analysis: Dict[str, str]

class FullPipelineRequest(BaseModel):
    job_description: str
    fairness_weight: float = 0.5
    time_weight: float = 0.3
    max_duration: int = 90

class FullPipelineResponse(BaseModel):
    skills: Dict[str, float]
    recommendation: BatteryRecommendation
    explanation: Dict[str, any]
    alternatives: List[BatteryRecommendation]
    counterfactuals: List[Dict[str, any]]

# Initialize core components
catalog_path = project_root / "data" / "processed" / "assessment_catalog.json"
generator = BatteryGenerator(str(catalog_path), max_battery_size=3)
fairness_analyzer = FairnessAnalyzer(str(catalog_path))
tracer = DecisionTracer(str(catalog_path))

# Initialize RAG engine
if RAG_AVAILABLE:
    rag_engine = RAGEngine(str(catalog_path))
else:
    rag_engine = None

@app.get("/")
def root():
    """API root endpoint."""
    return {
        "name": "Assessment Recommendation Engine API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "extract_skills": "/api/extract-skills",
            "simulate_performance": "/api/simulate-performance",
            "optimize_battery": "/api/optimize-battery",
            "full_pipeline": "/api/full-pipeline"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "components": "loaded"}

@app.post("/api/extract-skills", response_model=SkillExtractionResponse)
def extract_skills(request: SkillExtractionRequest):
    """
    Phase 2: Extract skill distribution from job description.
    
    This endpoint demonstrates probabilistic skill extraction with uncertainty.
    """
    try:
        # Simple keyword-based extraction (Phase 2 implementation)
        skills = _extract_skills_from_text(request.job_description)
        
        # Compute entropy
        import math
        entropy = -sum(p * math.log2(p) for p in skills.values() if p > 0)
        
        # Get top skills
        top_skills = [
            {"skill": k, "weight": v} 
            for k, v in sorted(skills.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        return SkillExtractionResponse(
            skills=skills,
            entropy=entropy,
            top_skills=top_skills
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate-performance", response_model=PerformanceSimulationResponse)
def simulate_performance(request: PerformanceSimulationRequest):
    """
    Phase 3: Simulate performance using causal model.
    
    This endpoint demonstrates Skills → Assessments → Performance causal inference.
    """
    try:
        estimator = PerformanceEstimator(request.skills)
        
        # Create mock battery metadata
        battery_meta = {
            "battery_id": request.battery_id,
            "assessments": request.battery_id.split("_"),
            "size": len(request.battery_id.split("_")),
            "total_duration": 45
        }
        
        result = estimator.estimate_battery_performance(battery_meta)
        
        return PerformanceSimulationResponse(
            expected_performance=result["expected_performance"],
            uncertainty=result["performance_std"],
            skill_coverage=result["skill_coverage"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize-battery", response_model=OptimizationResponse)
def optimize_battery(request: OptimizationRequest):
    """
    Phase 4: Multi-objective battery optimization.
    
    This endpoint demonstrates the complete optimization engine with trade-offs.
    """
    try:
        # Generate and evaluate batteries
        constraint_manager = ConstraintManager(max_duration=request.max_duration)
        batteries = generator.generate_batteries_with_metadata()
        feasible_batteries = constraint_manager.filter_feasible_batteries(batteries)
        
        # Evaluate performance and fairness
        estimator = PerformanceEstimator(request.skills)
        evaluated_batteries = []
        
        for battery in feasible_batteries[:50]:  # Limit for API performance
            perf_est = estimator.estimate_battery_performance(battery)
            battery.update(perf_est)
            
            fairness_est = fairness_analyzer.compute_battery_fairness_risk(battery)
            battery.update(fairness_est)
            
            evaluated_batteries.append(battery)
        
        # Optimize
        weights = UtilityWeights(
            alpha=1.0,
            beta=request.fairness_weight,
            gamma=request.time_weight,
            delta=0.2
        )
        
        optimizer = UtilityOptimizer(weights)
        ranked = optimizer.rank_batteries(evaluated_batteries)
        
        if not ranked:
            raise HTTPException(status_code=404, detail="No valid batteries found")
        
        # Format response
        top = ranked[0]
        top_rec = BatteryRecommendation(
            battery_id=top["battery_id"],
            assessments=top["assessments"],
            expected_performance=top["expected_performance"],
            fairness_risk=top["total_fairness_risk"],
            duration=top["total_duration"],
            utility=top["total_utility"]
        )
        
        alternatives = [
            BatteryRecommendation(
                battery_id=b["battery_id"],
                assessments=b["assessments"],
                expected_performance=b["expected_performance"],
                fairness_risk=b["total_fairness_risk"],
                duration=b["total_duration"],
                utility=b["total_utility"]
            )
            for b in ranked[1:6]
        ]
        
        # Pareto frontier
        pareto = optimizer.find_pareto_frontier(evaluated_batteries)
        pareto_frontier = [
            {
                "performance": b["expected_performance"],
                "fairness_risk": b["total_fairness_risk"],
                "duration": b["total_duration"]
            }
            for b in pareto[:10]
        ]
        
        # Trade-off analysis
        trade_offs = {
            "performance_fairness": f"Correlation: {_compute_correlation(ranked):.3f}",
            "pareto_count": f"{len(pareto)} efficient solutions",
            "top_utility": f"{top['total_utility']:.3f}"
        }
        
        return OptimizationResponse(
            top_recommendation=top_rec,
            alternatives=alternatives,
            pareto_frontier=pareto_frontier,
            trade_off_analysis=trade_offs
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag-recommend")
def rag_recommend(request: FullPipelineRequest):
    """
    RAG-based recommendation endpoint.
    
    Uses retrieval-augmented generation for intelligent recommendations.
    Returns JSON response with recommendations and explanations.
    """
    if not RAG_AVAILABLE or rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not available")
    
    try:
        # Generate recommendation using RAG
        result = rag_engine.generate_recommendation(
            query=request.job_description,
            preferences={
                "fairness_weight": request.fairness_weight,
                "time_weight": request.time_weight,
                "max_duration": request.max_duration
            }
        )
        
        return {
            "success": True,
            "query": request.job_description,
            "recommendation": result.get("top_recommendation"),
            "alternatives": result.get("alternatives", []),
            "explanation": result.get("explanation"),
            "retrieval_results": result.get("retrieval_results", []),
            "method": "RAG"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/full-pipeline", response_model=FullPipelineResponse)
def full_pipeline(request: FullPipelineRequest):
    """
    Complete end-to-end pipeline: JD → Skills → Optimization → Explanation.
    
    This is the main demo endpoint that showcases the entire system.
    """
    try:
        # Phase 2: Extract skills
        skills = _extract_skills_from_text(request.job_description)
        
        # Phase 4: Optimize
        constraint_manager = ConstraintManager(max_duration=request.max_duration)
        batteries = generator.generate_batteries_with_metadata()
        feasible_batteries = constraint_manager.filter_feasible_batteries(batteries)
        
        estimator = PerformanceEstimator(skills)
        evaluated_batteries = []
        
        for battery in feasible_batteries[:30]:
            perf_est = estimator.estimate_battery_performance(battery)
            battery.update(perf_est)
            
            fairness_est = fairness_analyzer.compute_battery_fairness_risk(battery)
            battery.update(fairness_est)
            
            evaluated_batteries.append(battery)
        
        weights = UtilityWeights(
            alpha=1.0,
            beta=request.fairness_weight,
            gamma=request.time_weight,
            delta=0.2
        )
        
        optimizer = UtilityOptimizer(weights)
        ranked = optimizer.rank_batteries(evaluated_batteries)
        
        if not ranked:
            raise HTTPException(status_code=404, detail="No valid batteries found")
        
        top = ranked[0]
        
        # Phase 5: Generate explanation
        utility_components = {
            "components": {
                "performance": top["expected_performance"],
                "fairness": -top["total_fairness_risk"] * request.fairness_weight,
                "time": -top["total_duration"] * request.time_weight / 100
            }
        }
        
        decision_trace = tracer.generate_decision_trace(
            top, skills, utility_components, ranked[1:4]
        )
        
        # Format response
        recommendation = BatteryRecommendation(
            battery_id=top["battery_id"],
            assessments=top["assessments"],
            expected_performance=top["expected_performance"],
            fairness_risk=top["total_fairness_risk"],
            duration=top["total_duration"],
            utility=top["total_utility"]
        )
        
        alternatives = [
            BatteryRecommendation(
                battery_id=b["battery_id"],
                assessments=b["assessments"],
                expected_performance=b["expected_performance"],
                fairness_risk=b["total_fairness_risk"],
                duration=b["total_duration"],
                utility=b["total_utility"]
            )
            for b in ranked[1:4]
        ]
        
        # Counterfactuals
        counterfactuals = [
            {
                "scenario": "High Fairness Priority",
                "fairness_weight": 1.0,
                "impact": "Would select lower-risk battery"
            },
            {
                "scenario": "Time Constrained",
                "max_duration": 45,
                "impact": "Would select shorter battery"
            }
        ]
        
        return FullPipelineResponse(
            skills=skills,
            recommendation=recommendation,
            explanation=decision_trace,
            alternatives=alternatives,
            counterfactuals=counterfactuals
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
def _extract_skills_from_text(text: str) -> Dict[str, float]:
    """Simple skill extraction (Phase 2 implementation)."""
    skill_keywords = {
        "C1": ["analytical", "problem-solving", "reasoning", "logic"],
        "C2": ["processing", "speed", "efficiency", "quick"],
        "C4": ["spatial", "design", "technical"],
        "B1": ["reliable", "organized", "detail"],
        "B2": ["stable", "calm", "resilient"],
        "B4": ["team", "collaborate", "social"],
        "W1": ["achievement", "goal", "drive"],
        "W2": ["leadership", "lead", "manage"],
        "W4": ["communication", "interpersonal"],
        "W5": ["initiative", "proactive", "independent"]
    }
    
    text_lower = text.lower()
    skill_scores = {}
    
    for skill_id, keywords in skill_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        skill_scores[skill_id] = score
    
    total = sum(skill_scores.values())
    if total > 0:
        return {k: v/total for k, v in skill_scores.items() if v > 0}
    else:
        return {"C1": 0.3, "B4": 0.2, "W4": 0.2, "W5": 0.15, "B1": 0.15}

def _compute_correlation(batteries: List[Dict]) -> float:
    """Compute performance-fairness correlation."""
    if len(batteries) < 2:
        return 0.0
    
    perfs = [b["expected_performance"] for b in batteries]
    fairs = [b["total_fairness_risk"] for b in batteries]
    
    # Simple correlation
    mean_p = sum(perfs) / len(perfs)
    mean_f = sum(fairs) / len(fairs)
    
    num = sum((p - mean_p) * (f - mean_f) for p, f in zip(perfs, fairs))
    den_p = sum((p - mean_p) ** 2 for p in perfs) ** 0.5
    den_f = sum((f - mean_f) ** 2 for f in fairs) ** 0.5
    
    if den_p == 0 or den_f == 0:
        return 0.0
    
    return num / (den_p * den_f)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)