```powershell
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

Local MCP Server (stdio)
```
 "research-server": {
        "command": "uv",
        "args": [
            "--directory", 
            "C:\\Users\\Usuario\\research-server", 
            "run", 
            "research_server.py"]
      }
```

Run the SSE server

```bash
uv run remote_research_server.py
````

Test it

```bash
npx @modelcontextprotocol/inspector
```

Add this to your `claude_desktop_config.json`(or Cursor)

```json
"remote-research-server": {
      "command": "npx",
      "args": ["mcp-remote", "http://localhost:8001/sse", "--allow-http"]
    },
```

In Windsurf, it's easier. Just add this:

```json
"remote-research-server-8001": {
      "serverUrl": "http://0.0.0.0:8001/sse"
    },
```
