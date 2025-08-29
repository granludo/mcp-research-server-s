#!/usr/bin/env python3
"""
Script to explore SerpAPI Google Scholar response structure
"""
import os
import sys
import requests
import json

# Add current directory to path to ensure we can import dotenv
sys.path.insert(0, os.getcwd())

# Import dotenv directly to avoid any other imports
try:
    from dotenv import load_dotenv
except ImportError:
    print("❌ python-dotenv not installed. Please install with: pip install python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

def explore_serpapi_response():
    """Make a test request to SerpAPI and explore the response structure"""

    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        print("❌ SERP_API_KEY not found in environment")
        return None

    # Test search parameters
    params = {
        "engine": "google_scholar",
        "q": "machine learning transformers",
        "api_key": api_key,
        "num": 3  # Just get a few results for exploration
    }

    print("🔍 Making test request to SerpAPI...")
    print(f"Search query: '{params['q']}'")
    print(f"Number of results: {params['num']}")

    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        results = response.json()

        print("\n✅ API Response received successfully")
        print(f"Response status: {response.status_code}")

        # Save full response for analysis
        with open('serpapi_response_sample.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("💾 Full response saved to 'serpapi_response_sample.json'")

        return results

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def analyze_response_structure(results):
    """Analyze the structure of the SerpAPI response"""

    if not results:
        return

    print("\n" + "="*60)
    print("📊 SERPAPI RESPONSE STRUCTURE ANALYSIS")
    print("="*60)

    # Top-level keys
    print("
🔑 Top-level response keys:"    for key in results.keys():
        print(f"  • {key}: {type(results[key])}")

    # Focus on organic_results
    if "organic_results" in results:
        organic = results["organic_results"]
        print(f"\n📈 Number of organic results: {len(organic)}")

        if organic:
            print("
🔍 First result detailed analysis:"            first_result = organic[0]
            analyze_single_result(first_result)

def analyze_single_result(result):
    """Analyze a single result to see all available fields"""

    print("Available fields in organic result:")
    for key, value in result.items():
        value_type = type(value).__name__
        if isinstance(value, (list, dict)):
            if isinstance(value, list):
                length = len(value)
                if value and isinstance(value[0], dict):
                    print(f"  • {key}: {value_type}[{length}] (objects)")
                else:
                    print(f"  • {key}: {value_type}[{length}] (values: {value})")
            else:
                sub_keys = list(value.keys())
                print(f"  • {key}: {value_type} with keys: {sub_keys}")
        else:
            # Truncate long strings for display
            str_value = str(value)
            if len(str_value) > 100:
                str_value = str_value[:97] + "..."
            print(f"  • {key}: {value_type} = {str_value}")

    # Special analysis for key fields
    print("
🔍 Key field analysis:"    fields_to_check = ['title', 'snippet', 'link', 'authors', 'publication_info', 'cited_by', 'resources']

    for field in fields_to_check:
        if field in result:
            value = result[field]
            print(f"\n  📋 {field.upper()}:")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    print(f"    • {sub_key}: {sub_value}")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        print(f"    [{i}] {item}")
                    else:
                        print(f"    [{i}] {item}")
            else:
                print(f"    {value}")
        else:
            print(f"\n  📋 {field.upper()}: NOT FOUND")

def main():
    print("🚀 SerpAPI Google Scholar Response Explorer")
    print("="*50)

    results = explore_serpapi_response()
    if results:
        analyze_response_structure(results)

        print("
" + "="*60)
        print("📝 SUMMARY")
        print("="*60)
        print("✅ Full response saved to 'serpapi_response_sample.json'")
        print("📊 Response structure analyzed above")
        print("🔍 Check the JSON file for complete data structure")
    else:
        print("❌ Failed to get response from SerpAPI")

if __name__ == "__main__":
    main()
