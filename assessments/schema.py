"""
Assessment schema definitions and validation using Pydantic.
Research-aware schema with explicit measurement properties.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Literal
from enum import Enum
import json


class AssessmentType(str, Enum):
    """Assessment types based on measurement approach."""
    COGNITIVE = "cognitive"
    PERSONALITY = "personality" 
    SITUATIONAL_JUDGMENT = "situational_judgment"
    SKILLS_BASED = "skills_based"
    BEHAVIORAL_INTERVIEW = "behavioral_interview"


class Assessment(BaseModel):
    """
    Research-aware assessment schema with explicit psychometric properties.
    
    This schema acknowledges measurement error, reliability constraints,
    and adverse impact considerations - critical for causal modeling.
    """
    assessment_id: str = Field(..., description="Unique identifier for the assessment")
    name: str = Field(..., description="Human-readable assessment name")
    assessment_type: AssessmentType = Field(..., description="Type of assessment methodology")
    
    # Core psychometric properties
    measured_constructs: Dict[str, float] = Field(
        ..., 
        description="Latent skills measured with loading coefficients (0.0-1.0)"
    )
    reliability: float = Field(
        ..., 
        ge=0.0, 
        lt=1.0,  # Always < 1.0 - perfect reliability doesn't exist
        description="Internal consistency reliability (Cronbach's alpha or similar)"
    )
    measurement_noise: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of variance due to measurement error (1 - reliability)"
    )
    
    # Practical constraints
    duration_minutes: int = Field(..., gt=0, description="Assessment duration in minutes")
    adverse_impact_risk: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Risk of adverse impact on protected groups (0=low, 1=high)"
    )
    
    # Optional metadata
    description: Optional[str] = Field(None, description="Detailed assessment description")
    provider: Optional[str] = Field(None, description="Assessment provider/vendor")
    cost_per_administration: Optional[float] = Field(None, ge=0, description="Cost per candidate")
    
    @validator('measurement_noise')
    def validate_measurement_noise(cls, v, values):
        """Ensure measurement noise is consistent with reliability."""
        if 'reliability' in values:
            expected_noise = 1.0 - values['reliability']
            if abs(v - expected_noise) > 0.05:  # Allow small tolerance
                raise ValueError(f"Measurement noise ({v}) inconsistent with reliability ({values['reliability']})")
        return v
    
    @validator('measured_constructs')
    def validate_construct_loadings(cls, v):
        """Ensure construct loadings are valid and assessment measures multiple constructs."""
        if not v:
            raise ValueError("Assessment must measure at least one construct")
        
        if len(v) == 1:
            raise ValueError("Assessment should measure multiple constructs for realistic modeling")
            
        for construct_id, loading in v.items():
            if not (0.0 <= loading <= 1.0):
                raise ValueError(f"Construct loading for {construct_id} must be between 0.0 and 1.0")
        
        return v
    
    @validator('adverse_impact_risk')
    def validate_adverse_impact_by_type(cls, v, values):
        """Validate adverse impact risk is realistic for assessment type."""
        if 'assessment_type' in values:
            assessment_type = values['assessment_type']
            
            # Cognitive tests typically have higher adverse impact risk
            if assessment_type == AssessmentType.COGNITIVE and v < 0.2:
                raise ValueError("Cognitive assessments typically have higher adverse impact risk (≥0.2)")
            
            # Personality tests typically have lower adverse impact risk  
            if assessment_type == AssessmentType.PERSONALITY and v > 0.3:
                raise ValueError("Personality assessments typically have lower adverse impact risk (≤0.3)")
        
        return v

    def get_primary_construct(self) -> str:
        """Get the construct with highest loading."""
        return max(self.measured_constructs.items(), key=lambda x: x[1])[0]
    
    def get_construct_coverage(self) -> List[str]:
        """Get all constructs measured above threshold (0.3)."""
        return [construct for construct, loading in self.measured_constructs.items() if loading >= 0.3]


class JobProfile(BaseModel):
    """Job profile with required latent skills."""
    job_title: str = Field(..., description="Job title or role name")
    required_constructs: Dict[str, float] = Field(
        ..., 
        description="Required latent skills with importance weights"
    )
    experience_level: Literal["entry", "mid", "senior", "executive"] = Field(
        ..., 
        description="Required experience level"
    )
    department: Optional[str] = Field(None, description="Organizational department")
    
    @validator('required_constructs')
    def validate_construct_weights(cls, v):
        """Ensure construct weights sum to reasonable total."""
        if not v:
            raise ValueError("Job must require at least one construct")
        
        total_weight = sum(v.values())
        if not (0.8 <= total_weight <= 1.2):  # Allow some flexibility
            raise ValueError(f"Construct weights should sum to ~1.0, got {total_weight}")
        
        return v


class AssessmentRecommendation(BaseModel):
    """Assessment recommendation with transparent scoring."""
    assessment: Assessment
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Overall relevance to job")
    construct_alignment: Dict[str, float] = Field(
        ..., 
        description="Alignment between job requirements and assessment constructs"
    )
    efficiency_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Assessment efficiency (coverage per minute)"
    )
    risk_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Risk factors (adverse impact, measurement error, etc.)"
    )
    reasoning: str = Field(..., description="Human-readable explanation for recommendation")
    
    def get_overall_score(self) -> float:
        """Calculate overall recommendation score."""
        # Weighted combination of factors
        return (
            0.5 * self.relevance_score + 
            0.3 * self.efficiency_score + 
            0.2 * (1.0 - self.risk_factors.get('adverse_impact', 0.0))
        )


class AssessmentBattery(BaseModel):
    """Collection of assessments for comprehensive evaluation."""
    battery_id: str = Field(..., description="Unique identifier for assessment battery")
    assessments: List[Assessment] = Field(..., description="List of assessments in battery")
    total_duration: int = Field(..., description="Total duration in minutes")
    construct_coverage: Dict[str, float] = Field(
        ..., 
        description="Overall construct coverage across all assessments"
    )
    
    @validator('total_duration')
    def validate_total_duration(cls, v, values):
        """Ensure total duration matches sum of individual assessments."""
        if 'assessments' in values:
            calculated_duration = sum(a.duration_minutes for a in values['assessments'])
            if v != calculated_duration:
                raise ValueError(f"Total duration ({v}) doesn't match sum of assessments ({calculated_duration})")
        return v