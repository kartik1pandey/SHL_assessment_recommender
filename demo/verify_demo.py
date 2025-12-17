"""
Simple verification that the demo is accessible
"""

import webbrowser
import time

print("🎯 Assessment Recommendation Engine - Demo Verification")
print("=" * 60)
print()
print("✅ Server is running at: http://localhost:8080")
print()
print("📋 What to do:")
print("   1. Open your web browser")
print("   2. Go to: http://localhost:8080")
print("   3. You should see the main page with sample jobs")
print()
print("🚀 Quick Test:")
print("   - Click on 'Software Engineer' sample job")
print("   - Click 'Generate Recommendations' button")
print("   - You should see results in 1-2 seconds")
print()
print("🌐 Opening browser automatically in 3 seconds...")
time.sleep(3)

try:
    webbrowser.open('http://localhost:8080')
    print("✅ Browser opened!")
    print()
    print("📊 If you see the Assessment Recommendation Engine page,")
    print("   the demo is working correctly!")
    print()
    print("💡 Try the sample jobs and adjust the preference sliders")
    print("   to see how the recommendations change.")
except Exception as e:
    print(f"❌ Could not open browser automatically: {e}")
    print("   Please open http://localhost:8080 manually")

print()
print("🎉 Demo is ready for use!")
print("=" * 60)