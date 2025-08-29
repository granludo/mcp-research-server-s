#!/usr/bin/env python3
"""
Debug script to check environment variables and engine discovery.
"""

import os
import sys
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔍 Research MCP Server Environment Debug")
print("=" * 50)

print("\n📍 Environment Information:")
print(f"Current working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

print("\n🔑 Environment Variables:")
print(f"SERP_API_KEY set: {bool(os.getenv('SERP_API_KEY'))}")
if os.getenv('SERP_API_KEY'):
    print(f"SERP_API_KEY length: {len(os.getenv('SERP_API_KEY'))}")
    print(f"SERP_API_KEY preview: {os.getenv('SERP_API_KEY')[:10]}...")

print(f"SEMANTIC_SCHOLAR_API_KEY set: {bool(os.getenv('SEMANTIC_SCHOLAR_API_KEY'))}")
print(f"PUBMED_API_KEY set: {bool(os.getenv('PUBMED_API_KEY'))}")

# Show all API-related environment variables
api_vars = [k for k in os.environ.keys() if 'API' in k.upper()]
print(f"All API-related env vars: {api_vars}")

print("\n📁 File System Check:")
engines_dir = os.path.join(os.getcwd(), 'knowledge-base', 'engines')
print(f"Engines directory exists: {os.path.exists(engines_dir)}")
if os.path.exists(engines_dir):
    files = os.listdir(engines_dir)
    print(f"Files in engines directory: {files}")

    google_scholar_path = os.path.join(engines_dir, 'google_scholar.py')
    print(f"Google Scholar engine file exists: {os.path.exists(google_scholar_path)}")

print("\n🧪 Testing Config Loading:")
try:
    from config import Config
    config = Config()
    print("✅ Config loaded successfully")
    print(f"SERP_API_KEY from config: {bool(config.get_api_key('google_scholar'))}")
except Exception as e:
    print(f"❌ Config loading failed: {e}")

print("\n🧪 Testing Google Scholar Engine:")
try:
    # Add the engines directory to Python path
    engines_path = os.path.join(os.getcwd(), 'knowledge-base', 'engines')
    sys.path.insert(0, engines_path)

    import google_scholar
    engine = google_scholar.GoogleScholarEngine()
    print(f"✅ Google Scholar engine created: {engine.name}")
    print(f"Engine ID: {engine.id}")
    available = engine.is_available()
    print(f"Engine available: {available}")
    if not available:
        print("❌ Engine reports as not available - check SERP_API_KEY")
    else:
        print("✅ Engine is available!")
except Exception as e:
    print(f"❌ Google Scholar engine test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🎯 Next Steps:")
if not os.getenv('SERP_API_KEY'):
    print("1. Check if .env file exists in project root")
    print("2. Ensure .env file contains: SERP_API_KEY=your_key_here")
    print("3. Re-run this debug script")
    print("4. Test the MCP server again")
else:
    print("1. ✅ Environment looks good!")
    print("2. ✅ .env file is being loaded automatically")
    print("3. Run the MCP server with --verbose")
    print("4. Both Local and Google Scholar engines should be available")

print("\n" + "=" * 50)
print("Debug complete!")
