#!/usr/bin/env python3
"""
Test script for the /codefix/analyze endpoint.
Run the FastAPI server first: python main.py
Then run this script: python test_codefix.py
"""
import requests
import json

API_URL = "http://localhost:8080"

def test_codefix():
    """Test the codefix endpoint with a simple bug."""

    # Example 1: Division by zero bug
    test_case_1 = {
        "code": """
def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / count
""",
        "issue": "This function crashes when passed an empty list"
    }

    # Example 2: Off-by-one error
    test_case_2 = {
        "code": """
def get_last_element(arr):
    return arr[len(arr)]
""",
        "issue": "IndexError when accessing the last element"
    }

    # Example 3: Type error
    test_case_3 = {
        "code": """
def greet(name):
    return "Hello, " + name + "!"

result = greet(123)
""",
        "issue": "TypeError when passing a number instead of string"
    }

    test_cases = [test_case_1, test_case_2, test_case_3]

    print("Testing /codefix/analyze endpoint...")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Issue: {test['issue']}")
        print(f"\nOriginal Code:")
        print(test['code'])

        try:
            response = requests.post(
                f"{API_URL}/codefix/analyze",
                json=test,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                print(f"\nFixed Code:")
                print(result['fix'])
                print("\n" + "=" * 60)
            else:
                print(f"\nError: {response.status_code}")
                print(response.text)
                print("\n" + "=" * 60)

        except requests.exceptions.ConnectionError:
            print("\n❌ Could not connect to API. Make sure the server is running:")
            print("   python main.py")
            return
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return

def check_health():
    """Check if the API is running and model is loaded."""
    try:
        response = requests.get(f"{API_URL}/healthz", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✓ API is running")
            print(f"✓ Model loaded: {health['model_loaded']}")
            print(f"✓ Model ID: {health['model_id']}")
            print(f"✓ Device: {health['device']}")
            if health.get('last_error'):
                print(f"⚠ Last error: {health['last_error']}")
            print()
            return health['model_loaded']
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running:")
        print("   python main.py")
        return False
    except Exception as e:
        print(f"❌ Error checking health: {e}")
        return False

if __name__ == "__main__":
    print("CodeFix Endpoint Test\n")

    if check_health():
        test_codefix()
    else:
        print("\n⚠ Model is not loaded yet. The first API call will trigger loading.")
        print("This may take a while depending on your system.\n")
        test_codefix()
