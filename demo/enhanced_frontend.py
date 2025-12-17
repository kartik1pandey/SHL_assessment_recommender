"""
Enhanced Assessment Recommendation Engine Demo
Beautiful web interface with interactive visualizations
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

class EnhancedDemoHandler(BaseHTTPRequestHandler):
    """Enhanced HTTP request handler with visualizations."""
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path).path
        
        if parsed_path == '/' or parsed_path == '/index.html':
            self.serve_main_page()
        elif parsed_path == '/demo':
            self.serve_demo_page()
        elif parsed_path.startswith('/api/recommend'):
            self.handle_recommendation_api()
        else:
            self.send_error(404)
