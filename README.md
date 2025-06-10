run the SSE server

```bash
uv run remote_research_server.py
````

Test it

```bash
npx @modelcontextprotocol/inspector
```

Add this to your claude_desktop_config.json

```json
"remote-research-server": {
      "command": "npx",
      "args": ["mcp-remote", "http://localhost:8001/sse", "--allow-http"]
    },
```