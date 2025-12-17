"""
Simple launcher for the Assessment Recommendation Engine demo
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit demo."""
    print("🚀 Starting Assessment Recommendation Engine Demo...")
    print("=" * 50)
    
    # Check if we're in the right directory
    demo_dir = Path(__file__).parent
    app_file = demo_dir / "app.py"
    
    if not app_file.exists():
        print("❌ Error: app.py not found in demo directory")
        return
    
    # Launch Streamlit
    try:
        print("🌐 Launching web interface...")
        print("📍 URL will be: http://localhost:8501")
        print("🔧 Use Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_file),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        print("\n💡 Try running manually:")
        print(f"   streamlit run {app_file}")

if __name__ == "__main__":
    main()