"""
Quick test to verify the demo is working
"""

import urllib.request
import urllib.parse
import json
import time

def test_server():
    """Test if the server is responding."""
    try:
        print("🧪 Testing Assessment Recommendation Engine Demo")
        print("=" * 50)
        
        # Test 1: Check main page
        print("1. Testing main page...")
        response = urllib.request.urlopen("http://localhost:8080/", timeout=5)
        if response.status == 200:
            content = response.read().decode()
            if "Assessment Recommendation Engine" in content:
                print("   ✅ Main page loads correctly")
            else:
                print("   ❌ Main page content incorrect")
                return False
        else:
            print(f"   ❌ Main page returned status: {response.status}")
            return False
        
        # Test 2: Check demo page with sample job
        print("2. Testing demo page...")
        params = {
            'job_desc': 'Software Engineer with analytical skills and teamwork',
            'fairness': '0.5',
            'time': '0.3',
            'duration': '90'
        }
        
        query_string = urllib.parse.urlencode(params)
        demo_url = f"http://localhost:8080/demo?{query_string}"
        
        response = urllib.request.urlopen(demo_url, timeout=30)
        if response.status == 200:
            content = response.read().decode()
            if "Top Recommendation" in content and "battery_id" in content:
                print("   ✅ Demo page generates recommendations")
                
                # Extract some info from the response
                if "Expected Performance" in content:
                    print("   ✅ Performance metrics displayed")
                if "Fairness Risk" in content:
                    print("   ✅ Fairness metrics displayed")
                if "Primary Reason" in content:
                    print("   ✅ Explanations generated")
                    
            else:
                print("   ❌ Demo page doesn't show recommendations")
                return False
        else:
            print(f"   ❌ Demo page returned status: {response.status}")
            return False
        
        # Test 3: Check API endpoint
        print("3. Testing API endpoint...")
        api_data = {
            "job_desc": "Data Analyst with strong analytical and quantitative skills",
            "fairness": 0.5,
            "time": 0.3,
            "duration": 90
        }
        
        json_data = json.dumps(api_data).encode('utf-8')
        req = urllib.request.Request(
            "http://localhost:8080/api/recommend",
            data=json_data,
            headers={'Content-Type': 'application/json'}
        )
        
        response = urllib.request.urlopen(req, timeout=30)
        if response.status == 200:
            result = json.loads(response.read().decode())
            if result.get("success"):
                rec = result["recommendation"]
                print(f"   ✅ API working - Recommended: {rec['battery_id']}")
                print(f"   📊 Performance: {rec['performance']:.1%}")
                print(f"   ⚖️ Fairness Risk: {rec['fairness_risk']:.1%}")
                print(f"   ⏱️ Duration: {rec['duration']} minutes")
            else:
                print(f"   ❌ API returned error: {result.get('error', 'Unknown')}")
                return False
        else:
            print(f"   ❌ API returned status: {response.status}")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Demo is fully functional")
        print("🌐 Access at: http://localhost:8080")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    success = test_server()
    
    if success:
        print("\n🚀 DEMO IS READY!")
        print("Open your browser to: http://localhost:8080")
    else:
        print("\n⚠️ Demo has issues - check server logs")