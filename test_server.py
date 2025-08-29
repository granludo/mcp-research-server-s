#!/usr/bin/env python3
"""
Test script for the Research MCP Server.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ResearchMCPServer

async def test_server():
    """Test the server initialization and basic functionality."""
    print("Testing Research MCP Server...")

    try:
        # Create server instance
        server = ResearchMCPServer(verbose=True)

        # Initialize server
        print("Initializing server...")
        await server.initialize()

        # Test list_search_engines
        print("Testing list_search_engines...")
        result = server.list_search_engines()
        print("Search engines result:", result)

        print("✅ Server test completed successfully!")

    except Exception as e:
        print(f"❌ Server test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_server())
