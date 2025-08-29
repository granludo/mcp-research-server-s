#!/usr/bin/env python3
"""
Simple MCP client test for the Research MCP Server.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_mcp_stdio():
    """Test MCP server via stdio."""
    print("Testing MCP server via stdio...")

    # Import and start the server in a subprocess
    import subprocess
    import threading
    import time

    # Start the server in stdio mode
    server_process = subprocess.Popen(
        [sys.executable, "main.py", "--verbose"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    try:
        # Give the server time to start
        time.sleep(2)

        # Test list_tools request
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        }

        print("Sending tools/list request...")
        server_process.stdin.write(json.dumps(list_tools_request) + "\n")
        server_process.stdin.flush()

        # Read response
        response_line = server_process.stdout.readline()
        if response_line:
            try:
                response = json.loads(response_line.strip())
                print("✅ Server responded to tools/list:")
                print(json.dumps(response, indent=2))

                # Test calling list_search_engines
                if 'result' in response and 'tools' in response['result']:
                    tools = response['result']['tools']
                    list_engines_tool = None
                    for tool in tools:
                        if tool.get('name') == 'list_search_engines':
                            list_engines_tool = tool
                            break

                    if list_engines_tool:
                        print("\nTesting list_search_engines tool...")
                        call_tool_request = {
                            "jsonrpc": "2.0",
                            "id": 2,
                            "method": "tools/call",
                            "params": {
                                "name": "list_search_engines",
                                "arguments": {}
                            }
                        }

                        server_process.stdin.write(json.dumps(call_tool_request) + "\n")
                        server_process.stdin.flush()

                        # Read response
                        tool_response_line = server_process.stdout.readline()
                        if tool_response_line:
                            try:
                                tool_response = json.loads(tool_response_line.strip())
                                print("✅ Tool call response:")
                                print(json.dumps(tool_response, indent=2))
                            except json.JSONDecodeError as e:
                                print(f"❌ Failed to parse tool response: {e}")
                                print(f"Raw response: {tool_response_line}")
                        else:
                            print("❌ No response from tool call")
                    else:
                        print("❌ list_search_engines tool not found")

            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse server response: {e}")
                print(f"Raw response: {response_line}")
        else:
            print("❌ No response from server")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        # Print server stderr for debugging
        stderr_output = server_process.stderr.read()
        if stderr_output:
            print("Server stderr:")
            print(stderr_output)

    finally:
        # Clean up
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    asyncio.run(test_mcp_stdio())
