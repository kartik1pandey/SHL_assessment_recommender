"""
Simple Assessment Recommendation Engine Demo
A basic web interface using Python's built-in HTTP server
"""

import sys
import json
import html
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import webbrowser
import threading
import time

# Add project modules to path
demo_dir = Path(__file__).parent
project_root = demo_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'optimizer'))
sys.path.insert(0, str(project_root / 'explainability'))

# Import system components
try:
    from optimizer.battery_generator import BatteryGenerator
    from optimizer.performance_estimator import PerformanceEstimator
    from optimizer.fairness import FairnessAnalyzer
    from optimizer.constraints import ConstraintManager
    from optimizer.utility import UtilityOptimizer, UtilityWeights
    from explainability.decision_trace import DecisionTracer
    SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import system components: {e}")
    SYSTEM_AVAILABLE = False

# Import RAG components
try:
    from rag.rag_engine import RAGEngine
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import RAG components: {e}")
    RAG_AVAILABLE = False

class DemoHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the demo."""
    
    def do_GET(self):
        """Handle GET requests."""
        # Parse the path without query parameters
        parsed_path = urlparse(self.path).path
        
        if parsed_path == '/' or parsed_path == '/index.html':
            self.serve_main_page()
        elif parsed_path == '/demo':
            self.serve_demo_page()
        elif parsed_path.startswith('/api/recommend'):
            self.handle_recommendation_api()
        elif parsed_path.startswith('/api/rag-recommend'):
            self.handle_rag_recommendation_api()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/api/recommend':
            self.handle_recommendation_api()
        elif self.path == '/api/rag-recommend':
            self.handle_rag_recommendation_api()
        else:
            self.send_error(404)
    
    def serve_main_page(self):
        """Serve the main demo page."""
        html_content = self.get_main_html()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_demo_page(self):
        """Serve the demo results page."""
        # Parse query parameters
        query = urlparse(self.path).query
        params = parse_qs(query)
        
        # Get job description and preferences
        job_desc = params.get('job_desc', [''])[0]
        fairness_weight = float(params.get('fairness', ['0.5'])[0])
        time_weight = float(params.get('time', ['0.3'])[0])
        max_duration = int(params.get('duration', ['90'])[0])
        
        if job_desc and SYSTEM_AVAILABLE:
            results = self.run_recommendation(job_desc, fairness_weight, time_weight, max_duration)
            html_content = self.get_results_html(results, job_desc)
        else:
            html_content = self.get_error_html("No job description provided or system unavailable")
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def handle_recommendation_api(self):
        """Handle API requests for recommendations."""
        if not SYSTEM_AVAILABLE:
            self.send_json_response({"error": "System components not available"}, 500)
            return
        
        try:
            # Parse request data
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length).decode()
                data = json.loads(post_data)
            else:
                # GET request with query parameters
                query = urlparse(self.path).query
                params = parse_qs(query)
                data = {k: v[0] for k, v in params.items()}
            
            job_desc = data.get('job_desc', '')
            fairness_weight = float(data.get('fairness', 0.5))
            time_weight = float(data.get('time', 0.3))
            max_duration = int(data.get('duration', 90))
            
            if not job_desc:
                self.send_json_response({"error": "Job description required"}, 400)
                return
            
            results = self.run_recommendation(job_desc, fairness_weight, time_weight, max_duration)
            self.send_json_response(results)
            
        except Exception as e:
            self.send_json_response({"error": str(e)}, 500)
    
    def handle_rag_recommendation_api(self):
        """Handle RAG API requests for recommendations."""
        if not RAG_AVAILABLE:
            self.send_json_response({"error": "RAG engine not available"}, 503)
            return
        
        try:
            # Parse request data
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length).decode()
                data = json.loads(post_data)
            else:
                # GET request with query parameters
                query = urlparse(self.path).query
                params = parse_qs(query)
                data = {k: v[0] for k, v in params.items()}
            
            job_desc = data.get('job_description', data.get('job_desc', ''))
            fairness_weight = float(data.get('fairness_weight', data.get('fairness', 0.5)))
            time_weight = float(data.get('time_weight', data.get('time', 0.3)))
            max_duration = int(data.get('max_duration', data.get('duration', 90)))
            
            if not job_desc:
                self.send_json_response({"error": "Job description required"}, 400)
                return
            
            results = self.run_rag_recommendation(job_desc, fairness_weight, time_weight, max_duration)
            self.send_json_response(results)
            
        except Exception as e:
            self.send_json_response({"error": str(e)}, 500)
    
    def send_json_response(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def run_recommendation(self, job_description, fairness_weight, time_weight, max_duration):
        """Run the recommendation pipeline."""
        try:
            # Get absolute path to data file
            demo_dir = Path(__file__).parent
            project_root = demo_dir.parent
            catalog_path = project_root / "data" / "processed" / "assessment_catalog.json"
            
            # Initialize components
            generator = BatteryGenerator(str(catalog_path), max_battery_size=3)
            fairness_analyzer = FairnessAnalyzer(str(catalog_path))
            constraint_manager = ConstraintManager(max_duration=max_duration)
            tracer = DecisionTracer(str(catalog_path))
            
            # Extract skills (simplified)
            job_skills = self.extract_skills_simple(job_description)
            
            # Generate batteries
            batteries = generator.generate_batteries_with_metadata()
            feasible_batteries = constraint_manager.filter_feasible_batteries(batteries)
            
            if not feasible_batteries:
                return {"error": "No feasible batteries found"}
            
            # Evaluate batteries
            estimator = PerformanceEstimator(job_skills)
            evaluated_batteries = []
            
            for battery in feasible_batteries[:20]:  # Limit for demo
                perf_est = estimator.estimate_battery_performance(battery)
                battery.update(perf_est)
                
                fairness_est = fairness_analyzer.compute_battery_fairness_risk(battery)
                battery.update(fairness_est)
                
                evaluated_batteries.append(battery)
            
            # Optimize
            weights = UtilityWeights(alpha=1.0, beta=fairness_weight, gamma=time_weight, delta=0.2)
            optimizer = UtilityOptimizer(weights)
            ranked = optimizer.rank_batteries(evaluated_batteries)
            
            if not ranked:
                return {"error": "No valid recommendations"}
            
            # Generate explanation
            top_battery = ranked[0]
            utility_components = {
                "components": {
                    "performance": top_battery.get("expected_performance", 0),
                    "fairness": -top_battery.get("total_fairness_risk", 0) * fairness_weight,
                    "time": -top_battery.get("total_duration", 0) * time_weight / 100
                }
            }
            
            decision_trace = tracer.generate_decision_trace(
                top_battery, job_skills, utility_components, ranked[1:4]
            )
            
            return {
                "success": True,
                "recommendation": {
                    "battery_id": top_battery["battery_id"],
                    "assessments": top_battery["assessments"],
                    "duration": top_battery["total_duration"],
                    "performance": top_battery["expected_performance"],
                    "fairness_risk": top_battery["total_fairness_risk"],
                    "utility": top_battery["total_utility"]
                },
                "explanation": {
                    "primary_reason": decision_trace["rationale"]["primary_reason"],
                    "key_strengths": decision_trace["rationale"]["key_strengths"],
                    "trade_offs": decision_trace["rationale"]["main_trade_offs"]
                },
                "alternatives": [
                    {
                        "battery_id": alt["battery_id"],
                        "performance": alt["expected_performance"],
                        "fairness_risk": alt["total_fairness_risk"],
                        "duration": alt["total_duration"]
                    }
                    for alt in ranked[1:4]
                ],
                "job_skills": job_skills
            }
            
        except Exception as e:
            return {"error": f"Recommendation failed: {str(e)}"}
    
    def run_rag_recommendation(self, job_description, fairness_weight, time_weight, max_duration):
        """Run RAG-based recommendation."""
        try:
            demo_dir = Path(__file__).parent
            project_root = demo_dir.parent
            catalog_path = project_root / "data" / "processed" / "assessment_catalog.json"
            
            # Initialize RAG engine
            rag_engine = RAGEngine(str(catalog_path))
            
            # Generate recommendation
            result = rag_engine.generate_recommendation(
                query=job_description,
                preferences={
                    "fairness_weight": fairness_weight,
                    "time_weight": time_weight,
                    "max_duration": max_duration
                }
            )
            
            # Format response
            top_rec = result.get("top_recommendation", {})
            
            return {
                "success": True,
                "query": job_description,
                "recommendation": {
                    "assessment_id": top_rec.get("assessment_id", ""),
                    "name": top_rec.get("name", ""),
                    "recommendation_score": top_rec.get("recommendation_score", 0.0),
                    "validity": top_rec.get("validity", 0.0),
                    "fairness_metrics": top_rec.get("fairness_metrics", {}),
                    "duration_minutes": top_rec.get("duration_minutes", 0)
                },
                "alternatives": result.get("alternatives", [])[:3],
                "explanation": result.get("explanation", {}),
                "retrieval_results": result.get("retrieval_results", []),
                "method": "RAG"
            }
            
        except Exception as e:
            return {"error": f"RAG recommendation failed: {str(e)}"}
    
    def extract_skills_simple(self, job_description):
        """Simple skill extraction."""
        skill_keywords = {
            "C1": ["analytical", "problem-solving", "reasoning", "logic"],
            "C2": ["processing", "speed", "efficiency", "quick"],
            "C4": ["spatial", "design", "technical", "visual"],
            "B1": ["reliable", "organized", "detail", "accuracy"],
            "B2": ["stable", "calm", "resilient", "stress"],
            "B4": ["team", "collaborate", "group", "social"],
            "W1": ["achievement", "goal", "drive", "performance"],
            "W2": ["leadership", "lead", "manage", "direct"],
            "W4": ["communication", "interpersonal", "presentation"],
            "W5": ["initiative", "proactive", "independent"]
        }
        
        jd_lower = job_description.lower()
        skill_scores = {}
        
        for skill_id, keywords in skill_keywords.items():
            score = sum(1 for keyword in keywords if keyword in jd_lower)
            skill_scores[skill_id] = score
        
        total_score = sum(skill_scores.values())
        if total_score > 0:
            return {k: v/total_score for k, v in skill_scores.items() if v > 0}
        else:
            return {"C1": 0.3, "B4": 0.2, "W4": 0.2, "W5": 0.15, "B1": 0.15}
    
    def get_main_html(self):
        """Generate main page HTML."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Assessment Recommendation Engine</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .header {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        .form-group {
            margin-bottom: 1rem;
        }
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: bold;
        }
        textarea, input, select {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 1rem;
        }
        textarea {
            height: 200px;
            resize: vertical;
        }
        .slider-container {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .slider {
            flex: 1;
        }
        .slider-value {
            min-width: 60px;
            text-align: center;
            font-weight: bold;
        }
        .btn {
            background: #1f77b4;
            color: white;
            padding: 1rem 2rem;
            border: none;
            border-radius: 4px;
            font-size: 1.1rem;
            cursor: pointer;
            width: 100%;
        }
        .btn:hover {
            background: #1565c0;
        }
        .sample-jobs {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .sample-job {
            background: #e3f2fd;
            padding: 1rem;
            border-radius: 4px;
            cursor: pointer;
            border: 2px solid transparent;
        }
        .sample-job:hover {
            border-color: #1f77b4;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }
        .feature {
            text-align: center;
            padding: 1rem;
        }
        .feature-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Assessment Recommendation Engine</h1>
        <p>Intelligent, Fair, and Explainable Assessment Selection</p>
    </div>

    <div class="card">
        <h2>Welcome! 👋</h2>
        <p>This system uses advanced multi-objective optimization to recommend the best assessment batteries for your job requirements.</p>
        
        <div class="features">
            <div class="feature">
                <div class="feature-icon">🧠</div>
                <h3>Scientifically Validated</h3>
                <p>Based on rigorous research and testing</p>
            </div>
            <div class="feature">
                <div class="feature-icon">⚖️</div>
                <h3>Fairness-Aware</h3>
                <p>Explicitly models and minimizes bias risk</p>
            </div>
            <div class="feature">
                <div class="feature-icon">💡</div>
                <h3>Fully Explainable</h3>
                <p>Every decision is transparent and justified</p>
            </div>
            <div class="feature">
                <div class="feature-icon">🎯</div>
                <h3>Stakeholder-Driven</h3>
                <p>You control the trade-offs that matter</p>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Try Sample Jobs</h2>
        <div class="sample-jobs">
            <div class="sample-job" onclick="loadSampleJob('software')">
                <h3>💻 Software Engineer</h3>
                <p>Full-stack developer with strong analytical skills</p>
            </div>
            <div class="sample-job" onclick="loadSampleJob('sales')">
                <h3>📈 Sales Manager</h3>
                <p>Regional territory leader with team management</p>
            </div>
            <div class="sample-job" onclick="loadSampleJob('analyst')">
                <h3>📊 Data Analyst</h3>
                <p>Business intelligence with quantitative focus</p>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>🔧 Configuration</h2>
        <form id="recommendationForm" onsubmit="submitForm(event)">
            <div class="form-group">
                <label for="jobDesc">Job Description:</label>
                <textarea id="jobDesc" name="job_desc" placeholder="Enter a detailed job description..." required></textarea>
            </div>

            <div class="form-group">
                <label for="fairness">Fairness Priority:</label>
                <div class="slider-container">
                    <input type="range" id="fairness" name="fairness" min="0" max="2" step="0.1" value="0.5" class="slider" oninput="updateSliderValue('fairness')">
                    <span id="fairnessValue" class="slider-value">0.5</span>
                </div>
                <small>Higher values prioritize fairness over performance</small>
            </div>

            <div class="form-group">
                <label for="time">Time Efficiency Priority:</label>
                <div class="slider-container">
                    <input type="range" id="time" name="time" min="0" max="1" step="0.1" value="0.3" class="slider" oninput="updateSliderValue('time')">
                    <span id="timeValue" class="slider-value">0.3</span>
                </div>
                <small>Higher values prioritize shorter assessments</small>
            </div>

            <div class="form-group">
                <label for="duration">Maximum Duration (minutes):</label>
                <div class="slider-container">
                    <input type="range" id="duration" name="duration" min="30" max="120" step="15" value="90" class="slider" oninput="updateSliderValue('duration')">
                    <span id="durationValue" class="slider-value">90</span>
                </div>
                <small>Hard limit on total assessment time</small>
            </div>

            <button type="submit" class="btn">🚀 Generate Recommendations</button>
        </form>
    </div>

    <script>
        const sampleJobs = {
            software: `Software Engineer - Full Stack Developer

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
- Ability to work independently and as part of a team`,

            sales: `Sales Manager - Regional Territory

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
- Goal-oriented with strong achievement drive`,

            analyst: `Data Analyst - Business Intelligence

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
- Ability to work independently on multiple projects`
        };

        function loadSampleJob(jobType) {
            document.getElementById('jobDesc').value = sampleJobs[jobType];
        }

        function updateSliderValue(sliderId) {
            const slider = document.getElementById(sliderId);
            const valueSpan = document.getElementById(sliderId + 'Value');
            valueSpan.textContent = slider.value;
        }

        function submitForm(event) {
            event.preventDefault();
            
            const formData = new FormData(event.target);
            const params = new URLSearchParams();
            
            for (let [key, value] of formData.entries()) {
                params.append(key, value);
            }
            
            // Show loading state
            const submitBtn = event.target.querySelector('button[type="submit"]');
            const originalText = submitBtn.textContent;
            submitBtn.textContent = '⏳ Generating...';
            submitBtn.disabled = true;
            
            // Navigate to results page
            window.location.href = '/demo?' + params.toString();
        }
    </script>
</body>
</html>
        """
    
    def get_results_html(self, results, job_description):
        """Generate results page HTML."""
        if "error" in results:
            return self.get_error_html(results["error"])
        
        rec = results["recommendation"]
        exp = results["explanation"]
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Assessment Recommendations</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }}
        .recommendation-box {{
            background-color: #e8f4fd;
            padding: 1.5rem;
            border-radius: 8px;
            border: 2px solid #1f77b4;
            margin: 1rem 0;
        }}
        .explanation-box {{
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #28a745;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #1f77b4;
        }}
        .alternatives {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }}
        .alternative {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            border-left: 4px solid #6c757d;
        }}
        .btn {{
            background: #6c757d;
            color: white;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 4px;
            text-decoration: none;
            display: inline-block;
            margin-top: 1rem;
        }}
        .btn:hover {{
            background: #5a6268;
        }}
        ul {{
            padding-left: 1.5rem;
        }}
        li {{
            margin-bottom: 0.5rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Assessment Recommendations</h1>
        <p>Results for your job requirements</p>
    </div>

    <div class="card">
        <h2>🏆 Top Recommendation</h2>
        <div class="recommendation-box">
            <h3>🎯 {rec['battery_id']}</h3>
            <p><strong>Assessments:</strong> {', '.join(rec['assessments'])}</p>
            <p><strong>Duration:</strong> {rec['duration']} minutes</p>
            <p><strong>Expected Performance:</strong> {rec['performance']:.1%}</p>
            <p><strong>Fairness Risk:</strong> {rec['fairness_risk']:.1%}</p>
            <p><strong>Utility Score:</strong> {rec['utility']:.3f}</p>
        </div>

        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{rec['performance']:.1%}</div>
                <div>Expected Performance</div>
            </div>
            <div class="metric">
                <div class="metric-value">{rec['fairness_risk']:.1%}</div>
                <div>Fairness Risk</div>
            </div>
            <div class="metric">
                <div class="metric-value">{rec['duration']}</div>
                <div>Total Minutes</div>
            </div>
            <div class="metric">
                <div class="metric-value">{rec['utility']:.3f}</div>
                <div>Utility Score</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>💡 Explanation</h2>
        <div class="explanation-box">
            <h4>Why This Battery?</h4>
            <p><strong>Primary Reason:</strong> {exp['primary_reason']}</p>
            
            <p><strong>Key Strengths:</strong></p>
            <ul>
                {''.join(f'<li>{strength}</li>' for strength in exp['key_strengths'])}
            </ul>
            
            <p><strong>Main Trade-offs:</strong></p>
            <ul>
                {''.join(f'<li>{tradeoff}</li>' for tradeoff in exp['trade_offs'])}
            </ul>
        </div>
    </div>

    <div class="card">
        <h2>🔄 Alternative Recommendations</h2>
        <div class="alternatives">
            {''.join(f'''
            <div class="alternative">
                <h4>{alt['battery_id']}</h4>
                <p><strong>Performance:</strong> {alt['performance']:.1%}</p>
                <p><strong>Fairness Risk:</strong> {alt['fairness_risk']:.1%}</p>
                <p><strong>Duration:</strong> {alt['duration']} min</p>
            </div>
            ''' for alt in results['alternatives'])}
        </div>
    </div>

    <div class="card">
        <h2>📊 Job Skills Analysis</h2>
        <p>Extracted skill requirements from your job description:</p>
        <div class="metrics">
            {''.join(f'''
            <div class="metric">
                <div class="metric-value">{weight:.1%}</div>
                <div>{skill}</div>
            </div>
            ''' for skill, weight in results['job_skills'].items())}
        </div>
    </div>

    <div class="card">
        <a href="/" class="btn">← Try Another Job</a>
    </div>
</body>
</html>
        """
    
    def get_error_html(self, error_message):
        """Generate error page HTML."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - Assessment Recommendation Engine</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .error-card {{
            background: white;
            border-radius: 8px;
            padding: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            border-left: 4px solid #dc3545;
        }}
        .btn {{
            background: #6c757d;
            color: white;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 4px;
            text-decoration: none;
            display: inline-block;
            margin-top: 1rem;
        }}
    </style>
</head>
<body>
    <div class="error-card">
        <h1>❌ Error</h1>
        <p>{html.escape(error_message)}</p>
        <a href="/" class="btn">← Back to Home</a>
    </div>
</body>
</html>
        """

def start_server(port=None, host='localhost', open_browser_flag=True):
    """Start the demo server."""
    import os
    
    # Use PORT from environment (for deployment) or default
    port = port or int(os.environ.get('PORT', 8080))
    
    # Use 0.0.0.0 for deployment, localhost for local
    if os.environ.get('PORT'):
        host = '0.0.0.0'
        open_browser_flag = False
    
    server_address = (host, port)
    httpd = HTTPServer(server_address, DemoHandler)
    
    print(f"🚀 Assessment Recommendation Engine Demo")
    print(f"=" * 50)
    print(f"🌐 Server running at: http://{host}:{port}")
    print(f"🔧 Press Ctrl+C to stop the server")
    print(f"📊 System components available: {SYSTEM_AVAILABLE}")
    print(f"-" * 50)
    
    # Open browser automatically (only for local development)
    if open_browser_flag:
        def open_browser():
            time.sleep(1)  # Wait for server to start
            webbrowser.open(f'http://localhost:{port}')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n👋 Server stopped by user")
        httpd.shutdown()

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else None
    start_server(port=port)