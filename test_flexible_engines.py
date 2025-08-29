#!/usr/bin/env python3
"""
Test script to verify flexible engines parameter handling.
"""

import json
import subprocess
import sys
import os

def test_flexible_engines():
    """Test different engines parameter formats."""
    print("🧪 Testing Flexible Engines Parameter")
    print("=" * 40)

    # Start the server
    server = subprocess.Popen(
        [sys.executable, 'main.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    try:
        # Initialize MCP protocol
        init_request = {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'initialize',
            'params': {
                'protocolVersion': '2024-11-05',
                'capabilities': {},
                'clientInfo': {'name': 'test', 'version': '1.0'}
            }
        }
        server.stdin.write(json.dumps(init_request) + '\n')
        server.stdin.flush()
        init_response = server.stdout.readline()

        # Send initialized notification
        init_notification = {'jsonrpc': '2.0', 'method': 'notifications/initialized'}
        server.stdin.write(json.dumps(init_notification) + '\n')
        server.stdin.flush()

        # Test 1: Single engine as string
        print("\n1️⃣ Testing single engine (string format)")
        search_request1 = {
            'jsonrpc': '2.0',
            'id': 2,
            'method': 'tools/call',
            'params': {
                'name': 'search_papers',
                'arguments': {
                    'query': 'quantum physics',
                    'engines': 'google_scholar',  # String format
                    'max_results': 1
                }
            }
        }
        server.stdin.write(json.dumps(search_request1) + '\n')
        server.stdin.flush()

        response1 = server.stdout.readline()
        result1 = json.loads(response1)
        if 'result' in result1 and 'content' in result1['result']:
            data1 = json.loads(result1['result']['content'][0]['text'])
            print(f"   ✅ String format: {data1['total_count']} results")
        else:
            print("   ❌ Error with string format")

        # Test 2: Multiple engines as array
        print("\n2️⃣ Testing multiple engines (array format)")
        search_request2 = {
            'jsonrpc': '2.0',
            'id': 3,
            'method': 'tools/call',
            'params': {
                'name': 'search_papers',
                'arguments': {
                    'query': 'machine learning',
                    'engines': ['google_scholar'],  # Array format
                    'max_results': 1
                }
            }
        }
        server.stdin.write(json.dumps(search_request2) + '\n')
        server.stdin.flush()

        response2 = server.stdout.readline()
        result2 = json.loads(response2)
        if 'result' in result2 and 'content' in result2['result']:
            data2 = json.loads(result2['result']['content'][0]['text'])
            print(f"   ✅ Array format: {data2['total_count']} results")
        else:
            print("   ❌ Error with array format")

        # Test 3: All engines (no engines parameter)
        print("\n3️⃣ Testing all engines (no engines parameter)")
        search_request3 = {
            'jsonrpc': '2.0',
            'id': 4,
            'method': 'tools/call',
            'params': {
                'name': 'search_papers',
                'arguments': {
                    'query': 'artificial intelligence',
                    'max_results': 1
                }
            }
        }
        server.stdin.write(json.dumps(search_request3) + '\n')
        server.stdin.flush()

        response3 = server.stdout.readline()
        result3 = json.loads(response3)
        if 'result' in result3 and 'content' in result3['result']:
            data3 = json.loads(result3['result']['content'][0]['text'])
            print(f"   ✅ All engines: {data3['total_count']} results")
        else:
            print("   ❌ Error with all engines")

        print("\n🎉 All flexible engines parameter formats working!")
        print("\n💡 MCP Inspector Usage:")
        print("   Single engine: engines='google_scholar'")
        print("   Multiple engines: engines=['google_scholar', 'local']")
        print("   All engines: omit engines parameter")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        server.terminate()

if __name__ == "__main__":
    test_flexible_engines()
