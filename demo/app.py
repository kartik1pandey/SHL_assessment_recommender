"""
Assessment Recommendation Engine - Web Demo
Interactive Streamlit application showcasing the complete system
"""

import streamlit as st
import sys
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List

# Add project modules to path
sys.path.append('../')
sys.path.append('../optimizer')
sys.path.append('../explainability')

# Import system components
from optimizer.battery_generator import BatteryGenerator
from optimizer.performance_estimator import PerformanceEstimator
from optimizer.fairness import FairnessAnalyzer
from optimizer.constraints import ConstraintManager
from optimizer.utility import UtilityOptimizer, UtilityWeights
from explainability.decision_trace import DecisionTracer

# Page configuration
st.set_page_config(
    page_title="Assessment Recommendation Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .explanation-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_system_components():
    """Load and cache system components."""
    try:
        generator = BatteryGenerator("../data/processed/assessment_catalog.json", max_battery_size=4)
        fairness_analyzer = FairnessAnalyzer("../data/processed/assessment_catalog.json")
        constraint_manager = ConstraintManager()
        tracer = DecisionTracer("../data/processed/assessment_catalog.json")
        
        return generator, fairness_analyzer, constraint_manager, tracer
    except Exception as e:
        st.error(f"Error loading system components: {e}")
        return None, None, None, None

@st.cache_data
def load_sample_jobs():
    """Load sample job descriptions."""
    return {
        "Software Engineer": {
            "description": """
            Software Engineer - Full Stack Developer
            
            We are seeking a skilled Software Engineer to join our development team. The ideal candidate will have strong problem-solving abilities, excellent communication skills, and the ability to work collaboratively in an agile environment.
            
            Key Responsibilities:
            - Design and develop scalable web applications
            - Write clean, maintainable code following best practices
            - Collaborate with cross-functional teams to deliver high-quality software
            - Participate in code reviews and technical discussions
            - Troubleshoot and debug complex technical issues
            
            Required Skills:
            - Strong analytical and problem-solving skills
            - Proficiency in multiple programming languages
            - Experience with database design and optimization
            - Excellent verbal and written communication
            - Ability to work independently and as part of a team
            """,
            "skills": {"C1": 0.25, "C2": 0.20, "C4": 0.15, "B1": 0.15, "W4": 0.10, "W5": 0.15}
        },
        "Sales Manager": {
            "description": """
            Sales Manager - Regional Territory
            
            We are looking for a dynamic Sales Manager to lead our regional sales team. The successful candidate will have strong leadership skills, excellent communication abilities, and a proven track record in sales performance.
            
            Key Responsibilities:
            - Lead and motivate a team of sales representatives
            - Develop and implement sales strategies
            - Build and maintain client relationships
            - Analyze sales data and market trends
            - Present to senior management and key clients
            
            Required Skills:
            - Exceptional leadership and team management abilities
            - Outstanding communication and presentation skills
            - Strong analytical and strategic thinking
            - Resilience and adaptability in fast-paced environment
            - Goal-oriented with strong achievement drive
            """,
            "skills": {"W2": 0.25, "W4": 0.20, "B2": 0.15, "W1": 0.15, "B4": 0.15, "C1": 0.10}
        },
        "Data Analyst": {
            "description": """
            Data Analyst - Business Intelligence
            
            We are seeking a detail-oriented Data Analyst to join our business intelligence team. The ideal candidate will have strong analytical skills, attention to detail, and the ability to translate complex data into actionable insights.
            
            Key Responsibilities:
            - Analyze large datasets to identify trends and patterns
            - Create reports and dashboards for stakeholders
            - Collaborate with business teams to understand requirements
            - Ensure data quality and accuracy
            - Present findings to management and recommend actions
            
            Required Skills:
            - Strong analytical and quantitative skills
            - Excellent attention to detail and accuracy
            - Proficiency in statistical analysis and data visualization
            - Clear communication of complex information
            - Ability to work independently on multiple projects
            """,
            "skills": {"C1": 0.30, "C2": 0.20, "B1": 0.20, "W4": 0.15, "C5": 0.15}
        }
    }

def extract_skills_from_jd(job_description: str) -> Dict[str, float]:
    """
    Simple skill extraction from job description.
    In production, this would use the Phase 2 extraction pipeline.
    """
    # Simplified keyword-based extraction
    skill_keywords = {
        "C1": ["analytical", "problem-solving", "reasoning", "logic", "thinking"],
        "C2": ["processing", "speed", "efficiency", "quick", "fast"],
        "C3": ["verbal", "language", "reading", "comprehension"],
        "C4": ["spatial", "design", "visual", "interface", "technical"],
        "C5": ["memory", "recall", "retention", "learning", "detail"],
        "B1": ["reliable", "consistent", "organized", "detail", "accuracy"],
        "B2": ["stable", "calm", "resilient", "stress", "pressure"],
        "B3": ["agreeable", "cooperative", "friendly", "supportive"],
        "B4": ["team", "collaborate", "group", "together", "social"],
        "B5": ["creative", "innovative", "open", "flexible", "adaptable"],
        "W1": ["achievement", "goal", "drive", "motivated", "performance"],
        "W2": ["leadership", "lead", "manage", "direct", "supervise"],
        "W3": ["adapt", "change", "flexible", "agile", "dynamic"],
        "W4": ["communication", "interpersonal", "presentation", "speaking"],
        "W5": ["initiative", "proactive", "self-directed", "independent"]
    }
    
    jd_lower = job_description.lower()
    skill_scores = {}
    
    for skill_id, keywords in skill_keywords.items():
        score = sum(1 for keyword in keywords if keyword in jd_lower)
        skill_scores[skill_id] = score
    
    # Normalize to probabilities
    total_score = sum(skill_scores.values())
    if total_score > 0:
        skill_distribution = {k: v/total_score for k, v in skill_scores.items() if v > 0}
    else:
        # Default distribution if no keywords found
        skill_distribution = {"C1": 0.3, "B4": 0.2, "W4": 0.2, "W5": 0.15, "B1": 0.15}
    
    return skill_distribution

def run_recommendation_pipeline(job_skills: Dict[str, float], 
                               fairness_weight: float,
                               time_weight: float,
                               max_duration: int,
                               generator, fairness_analyzer, constraint_manager, tracer):
    """Run the complete recommendation pipeline."""
    
    # Set up constraint manager with user preferences
    constraint_manager = ConstraintManager(max_duration=max_duration, ideal_duration=max_duration-15)
    
    # Generate candidate batteries
    batteries = generator.generate_batteries_with_metadata()
    feasible_batteries = constraint_manager.filter_feasible_batteries(batteries)
    
    if not feasible_batteries:
        return None, "No feasible batteries found with current constraints"
    
    # Evaluate performance
    estimator = PerformanceEstimator(job_skills)
    performance_results = []
    for battery in feasible_batteries:
        perf_est = estimator.estimate_battery_performance(battery)
        battery.update(perf_est)
        performance_results.append(battery)
    
    # Evaluate fairness
    complete_evaluations = []
    for battery in performance_results:
        fairness_est = fairness_analyzer.compute_battery_fairness_risk(battery)
        battery.update(fairness_est)
        complete_evaluations.append(battery)
    
    # Optimize with user preferences
    weights = UtilityWeights(
        alpha=1.0,  # Performance weight (fixed)
        beta=fairness_weight,
        gamma=time_weight,
        delta=0.2   # Constraint penalty (fixed)
    )
    
    optimizer = UtilityOptimizer(weights)
    ranked_batteries = optimizer.rank_batteries(complete_evaluations)
    
    if not ranked_batteries:
        return None, "No valid recommendations found"
    
    # Generate explanation for top recommendation
    top_battery = ranked_batteries[0]
    utility_components = {
        "components": {
            "performance": top_battery.get("expected_performance", 0),
            "fairness": -top_battery.get("total_fairness_risk", 0) * fairness_weight,
            "time": -top_battery.get("total_duration", 0) * time_weight / 100,
            "constraints": -top_battery.get("total_penalty", 0)
        }
    }
    
    decision_trace = tracer.generate_decision_trace(
        top_battery, job_skills, utility_components, ranked_batteries[1:4]
    )
    
    return {
        "top_recommendation": top_battery,
        "alternatives": ranked_batteries[1:6],  # Top 5 alternatives
        "all_evaluations": complete_evaluations,
        "decision_trace": decision_trace,
        "summary": tracer.generate_summary_explanation(decision_trace)
    }, None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Assessment Recommendation Engine</h1>', unsafe_allow_html=True)
    st.markdown("**Intelligent, Fair, and Explainable Assessment Selection**")
    
    # Load system components
    generator, fairness_analyzer, constraint_manager, tracer = load_system_components()
    
    if generator is None:
        st.error("Failed to load system components. Please check the file paths.")
        return
    
    # Sidebar for inputs
    st.sidebar.header("🔧 Configuration")
    
    # Job selection
    sample_jobs = load_sample_jobs()
    job_choice = st.sidebar.selectbox(
        "Select a sample job or use custom:",
        ["Custom"] + list(sample_jobs.keys())
    )
    
    if job_choice == "Custom":
        job_description = st.sidebar.text_area(
            "Job Description:",
            height=200,
            placeholder="Enter a detailed job description..."
        )
        if job_description:
            job_skills = extract_skills_from_jd(job_description)
        else:
            job_skills = {"C1": 0.3, "B1": 0.2, "W4": 0.2, "W5": 0.15, "B4": 0.15}
    else:
        job_description = sample_jobs[job_choice]["description"]
        job_skills = sample_jobs[job_choice]["skills"]
        st.sidebar.text_area("Job Description:", value=job_description, height=200, disabled=True)
    
    # Preference sliders
    st.sidebar.subheader("⚖️ Preferences")
    fairness_weight = st.sidebar.slider(
        "Fairness Priority",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Higher values prioritize fairness over performance"
    )
    
    time_weight = st.sidebar.slider(
        "Time Efficiency Priority",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Higher values prioritize shorter assessments"
    )
    
    max_duration = st.sidebar.slider(
        "Maximum Duration (minutes)",
        min_value=30,
        max_value=120,
        value=90,
        step=15,
        help="Hard limit on total assessment time"
    )
    
    # Run analysis button
    if st.sidebar.button("🚀 Generate Recommendations", type="primary"):
        with st.spinner("Analyzing job requirements and generating recommendations..."):
            results, error = run_recommendation_pipeline(
                job_skills, fairness_weight, time_weight, max_duration,
                generator, fairness_analyzer, constraint_manager, tracer
            )
        
        if error:
            st.error(f"Error: {error}")
            return
        
        # Store results in session state
        st.session_state.results = results
        st.session_state.job_skills = job_skills
        st.session_state.job_choice = job_choice
    
    # Display results if available
    if hasattr(st.session_state, 'results') and st.session_state.results:
        results = st.session_state.results
        job_skills = st.session_state.job_skills
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Top recommendation
            st.subheader("🏆 Top Recommendation")
            
            top_rec = results["top_recommendation"]
            
            st.markdown(f"""
            <div class="recommendation-box">
                <h3>🎯 {top_rec['battery_id']}</h3>
                <p><strong>Assessments:</strong> {', '.join(top_rec['assessments'])}</p>
                <p><strong>Duration:</strong> {top_rec['total_duration']} minutes</p>
                <p><strong>Expected Performance:</strong> {top_rec['expected_performance']:.1%}</p>
                <p><strong>Fairness Risk:</strong> {top_rec['total_fairness_risk']:.1%}</p>
                <p><strong>Utility Score:</strong> {top_rec['total_utility']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Explanation
            st.subheader("💡 Explanation")
            
            decision_trace = results["decision_trace"]
            
            st.markdown(f"""
            <div class="explanation-box">
                <h4>Why This Battery?</h4>
                <p><strong>Primary Reason:</strong> {decision_trace['rationale']['primary_reason']}</p>
                <p><strong>Key Strengths:</strong></p>
                <ul>
                    {''.join(f'<li>{strength}</li>' for strength in decision_trace['rationale']['key_strengths'])}
                </ul>
                <p><strong>Main Trade-offs:</strong></p>
                <ul>
                    {''.join(f'<li>{tradeoff}</li>' for tradeoff in decision_trace['rationale']['main_trade_offs'])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Skill coverage
            st.subheader("📊 Skill Coverage Analysis")
            
            skill_names = {
                "C1": "Fluid Intelligence", "C2": "Processing Speed", "C3": "Verbal Ability",
                "C4": "Spatial Ability", "C5": "Memory", "B1": "Conscientiousness",
                "B2": "Emotional Stability", "B3": "Agreeableness", "B4": "Teamwork",
                "B5": "Openness", "W1": "Achievement Focus", "W2": "Leadership",
                "W3": "Adaptability", "W4": "Communication", "W5": "Initiative"
            }
            
            coverage_data = []
            for skill_id, importance in job_skills.items():
                coverage = top_rec.get('skill_coverage', {}).get(skill_id, 0)
                coverage_data.append({
                    'Skill': skill_names.get(skill_id, skill_id),
                    'Job Importance': importance,
                    'Assessment Coverage': coverage,
                    'Gap': max(0, importance - coverage)
                })
            
            coverage_df = pd.DataFrame(coverage_data)
            
            fig_coverage = px.bar(
                coverage_df, 
                x='Skill', 
                y=['Job Importance', 'Assessment Coverage'],
                title="Job Requirements vs Assessment Coverage",
                barmode='group'
            )
            fig_coverage.update_layout(height=400)
            st.plotly_chart(fig_coverage, use_container_width=True)
        
        with col2:
            # Key metrics
            st.subheader("📈 Key Metrics")
            
            st.metric(
                "Expected Performance",
                f"{top_rec['expected_performance']:.1%}",
                delta=f"{(top_rec['expected_performance'] - 0.5):.1%} vs baseline"
            )
            
            st.metric(
                "Fairness Risk",
                f"{top_rec['total_fairness_risk']:.1%}",
                delta=f"{(0.5 - top_rec['total_fairness_risk']):.1%} vs average",
                delta_color="inverse"
            )
            
            st.metric(
                "Total Duration",
                f"{top_rec['total_duration']} min",
                delta=f"{max_duration - top_rec['total_duration']} min under limit"
            )
            
            st.metric(
                "Skill Coverage",
                f"{top_rec.get('coverage_score', 0):.1%}",
                delta="Comprehensive" if top_rec.get('coverage_score', 0) > 0.7 else "Adequate"
            )
            
            # Assessment details
            st.subheader("🔍 Assessment Details")
            
            assessment_details = decision_trace.get('assessment_details', {})
            for aid in top_rec['assessments']:
                if aid in assessment_details:
                    details = assessment_details[aid]
                    with st.expander(f"📋 {details['name']}"):
                        st.write(f"**Purpose:** {details['purpose']}")
                        st.write(f"**Duration:** {details['duration']}")
                        st.write(f"**Key Constructs:** {', '.join(details['key_constructs'])}")
                        st.write(f"**Selection Reason:** {details['selection_reason']}")
        
        # Alternative recommendations
        st.subheader("🔄 Alternative Recommendations")
        
        alternatives_data = []
        for i, alt in enumerate(results["alternatives"]):
            alternatives_data.append({
                'Rank': i + 2,
                'Battery': alt['battery_id'],
                'Performance': f"{alt['expected_performance']:.1%}",
                'Fairness Risk': f"{alt['total_fairness_risk']:.1%}",
                'Duration': f"{alt['total_duration']} min",
                'Utility': f"{alt['total_utility']:.3f}"
            })
        
        if alternatives_data:
            alt_df = pd.DataFrame(alternatives_data)
            st.dataframe(alt_df, use_container_width=True)
        
        # Trade-off visualization
        st.subheader("⚖️ Performance vs Fairness Trade-off")
        
        all_batteries = results["all_evaluations"]
        scatter_data = pd.DataFrame([{
            'Battery': b['battery_id'],
            'Performance': b['expected_performance'],
            'Fairness Risk': b['total_fairness_risk'],
            'Duration': b['total_duration'],
            'Type': 'Recommended' if b['battery_id'] == top_rec['battery_id'] else 'Alternative'
        } for b in all_batteries])
        
        fig_scatter = px.scatter(
            scatter_data,
            x='Fairness Risk',
            y='Performance',
            size='Duration',
            color='Type',
            hover_data=['Battery'],
            title="All Batteries: Performance vs Fairness Risk",
            color_discrete_map={'Recommended': 'red', 'Alternative': 'lightblue'}
        )
        
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Summary explanation
        st.subheader("📝 Executive Summary")
        st.markdown(f"""
        <div class="explanation-box">
            {results['summary']}
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the Assessment Recommendation Engine! 👋
        
        This system uses advanced multi-objective optimization to recommend the best assessment batteries for your job requirements.
        
        ### How it works:
        1. **📝 Job Analysis**: Extracts skill requirements from job descriptions
        2. **🧠 Causal Modeling**: Models relationships between skills, assessments, and performance
        3. **⚖️ Multi-Objective Optimization**: Balances performance, fairness, and efficiency
        4. **💡 Explainable Results**: Provides clear reasoning for every recommendation
        
        ### Get started:
        1. Select a sample job or enter your own job description in the sidebar
        2. Adjust your preferences for fairness and time efficiency
        3. Click "Generate Recommendations" to see the results!
        
        ### Key Features:
        - ✅ **Scientifically Validated**: Based on rigorous research and testing
        - ✅ **Fairness-Aware**: Explicitly models and minimizes bias risk
        - ✅ **Fully Explainable**: Every decision is transparent and justified
        - ✅ **Stakeholder-Driven**: You control the trade-offs that matter to you
        """)

if __name__ == "__main__":
    main()