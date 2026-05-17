#!/usr/bin/env python3
"""
smoke_test_web.py
=================
Automated smoke test for the LK20 Local Web Server.
Verifies core API connectivity and basic role-based access.
"""

import json
import urllib.request
import urllib.error
import time
import sys

BASE_URL = "http://127.0.0.1:8000/api"

def api_call(path, method="GET", data=None):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method=method)
    if data:
        req.add_header('Content-Type', 'application/json')
        req_data = json.dumps(data).encode('utf-8')
    else:
        req_data = None
        
    try:
        with urllib.request.urlopen(req, data=req_data) as response:
            return json.loads(response.read().decode('utf-8')), response.getcode()
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode('utf-8')), e.code
        except:
            return {"ok": False, "error": str(e)}, e.code
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

def run_tests():
    print(f"--- Starting Smoke Test: {BASE_URL} ---")
    
    # 0. Ensure clean state
    api_call("/logout", "POST")
    
    # 1. Health Check
    print("[1/6] Testing /health...", end=" ")
    res, code = api_call("/health")
    if code == 200 and res.get("ok"):
        print("OK")
    else:
        print(f"FAILED (Code: {code}, Error: {res.get('error')})")
        return

    # 2. Whoami (Guest)
    print("[2/6] Testing /whoami (Guest)...", end=" ")
    res, code = api_call("/whoami")
    if code == 200 and res.get("session", {}).get("role") == "guest":
        print("OK")
    else:
        print(f"FAILED (Role: {res.get('session', {}).get('role')})")

    # 3. Login as Admin
    print("[3/6] Testing /login (Admin)...", end=" ")
    login_data = {
        "role": "admin",
        "user_id": "smoke_tester",
        "school_org_id": "TEST-99"
    }
    res, code = api_call("/login", "POST", login_data)
    if code == 200 and res.get("session", {}).get("role") == "admin":
        print("OK")
    else:
        print(f"FAILED (Code: {code})")

    # 4. Create Network (Requires Admin)
    print("[4/6] Testing /create-network...", end=" ")
    res, code = api_call("/create-network", "POST")
    if code == 200 and res.get("ok"):
        print("OK")
    else:
        print(f"FAILED (Code: {code}, Error: {res.get('error')})")

    # 5. Gov Benefits
    print("[5/6] Testing /gov/benefits...", end=" ")
    res, code = api_call("/gov/benefits")
    if code == 200 and res.get("ok"):
        print("OK")
    else:
        print(f"FAILED (Code: {code})")

    # 6. AI Entropy Status
    print("[6/8] Testing /ai/entropy/status...", end=" ")
    res, code = api_call("/ai/entropy/status")
    if code == 200 and res.get("ok"):
        print("OK")
    else:
        print(f"FAILED (Code: {code}, Error: {res.get('error')})")

    # 7. AI Entropy Analyze
    print("[7/8] Testing /ai/entropy/analyze...", end=" ")
    analyze_data = {"text": "A cat gentles an animal.", "profile": "curriculum"}
    res, code = api_call("/ai/entropy/analyze", "POST", analyze_data)
    if code == 200 and not res.get("ok"): # Expected fail for "gentles"
        print("OK (Correctly flagged)")
    else:
        print(f"FAILED (Code: {code}, OK: {res.get('ok')})")

    # 8. Logout
    print("[8/8] Testing /logout...", end=" ")
    res, code = api_call("/logout", "POST")
    if code == 200 and res.get("ok"):
        print("OK")
    else:
        print(f"FAILED")


    print("\n--- Smoke Test Completed Successfully ---")

if __name__ == "__main__":
    # Give a tiny delay in case it's run immediately after server start
    time.sleep(1)
    try:
        run_tests()
    except KeyboardInterrupt:
        print("\nTest aborted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nCritical failure: {e}")
        sys.exit(1)
