#!/usr/bin/env python3
"""
Validation script for the Research MCP Server setup.
"""

import sys
import os
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def validate_imports():
    """Validate that all required modules can be imported."""
    print("🔍 Validating imports...")

    try:
        from config import Config
        print("✅ Config module imported successfully")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

    try:
        from database import DatabaseManager
        print("✅ Database module imported successfully")
    except ImportError as e:
        print(f"❌ Database import failed: {e}")
        return False

    try:
        from search_engines import SearchEngineManager, BaseSearchEngine
        print("✅ Search engines module imported successfully")
    except ImportError as e:
        print(f"❌ Search engines import failed: {e}")
        return False

    try:
        from main import ResearchMCPServer
        print("✅ Main module imported successfully")
    except ImportError as e:
        print(f"❌ Main import failed: {e}")
        return False

    return True

def validate_server_initialization():
    """Validate that the server can be initialized."""
    print("\n🔍 Validating server initialization...")

    try:
        from main import ResearchMCPServer
        import asyncio

        async def test_init():
            server = ResearchMCPServer(verbose=False)
            await server.initialize()
            return server

        # Run in a new event loop
        server = asyncio.run(test_init())

        # Test list_search_engines
        result = server.list_search_engines()
        print(f"✅ Server initialized successfully")
        print(f"✅ Search engines found: {result['total_count']}")
        for engine in result['engines']:
            print(f"  - {engine['name']} ({engine['id']}) - {'Available' if engine['is_available'] else 'Unavailable'}")

        return True

    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_environment():
    """Check environment configuration."""
    print("\n🔍 Checking environment configuration...")

    # Check for API keys
    serp_key = os.getenv('SERP_API_KEY')
    if serp_key:
        print("✅ SERP_API_KEY is set")
    else:
        print("⚠️  SERP_API_KEY not set (Google Scholar will be unavailable)")

    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python version {python_version.major}.{python_version.minor} is supported")
    else:
        print(f"❌ Python version {python_version.major}.{python_version.minor} is too old (need 3.8+)")

def main():
    """Main validation function."""
    print("🚀 Research MCP Server Validation")
    print("=" * 40)

    # Validate imports
    if not validate_imports():
        print("\n❌ Import validation failed. Please check your setup.")
        return False

    # Validate server initialization
    if not validate_server_initialization():
        print("\n❌ Server initialization failed. Please check your configuration.")
        return False

    # Check environment
    check_environment()

    print("\n" + "=" * 40)
    print("✅ Validation completed successfully!")
    print("\n📝 Next steps:")
    print("1. Set SERP_API_KEY environment variable for Google Scholar access")
    print("2. Run: python main.py --verbose")
    print("3. Test MCP tools using your MCP client")
    print("\n📖 Example MCP tool calls:")
    print("- list_search_engines: List available search engines")
    print("- search_papers: Search for academic papers")
    print("- create_project: Create a research project")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
