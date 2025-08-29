# Research MCP Server

A comprehensive research paper management system built as an MCP (Model Context Protocol) server. This server provides unified access to multiple academic search engines and maintains a local database of research papers with advanced organization capabilities.

## Features

- **Multi-Engine Search**: Search across Google Scholar, ArXiv, and local database
- **Local Database**: Store and organize research papers with projects and keywords
- **MCP Integration**: Full Model Context Protocol support for AI assistants
- **Keyword Management**: Manual keyword assignment and management
- **Citation Tracking**: Track citing papers and citation networks
- **Project Organization**: Organize papers into research projects
- **Extensible Architecture**: Easy to add new search engines

## Installation

1. **Clone or download** the repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables** (create `.env` file):
   ```bash
   # Required for Google Scholar
   SERP_API_KEY=your_serpapi_key_here

   # Optional: For additional search engines
   # SEMANTIC_SCHOLAR_API_KEY=your_key_here
   # PUBMED_API_KEY=your_key_here
   ```

## Quick Start

1. **Run the server**:
   ```bash
   python main.py --verbose
   ```

2. **Test basic functionality** (in another terminal):
   ```bash
   # List available search engines
   curl -X POST http://localhost:8000/jsonrpc \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc": "2.0", "id": 1, "method": "list_search_engines"}'

   # Search for papers
   curl -X POST http://localhost:8000/jsonrpc \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc": "2.0", "id": 2, "method": "search_papers", "params": {"query": "machine learning"}}'
   ```

## MCP Tools

### Core Tools
- **`list_search_engines`**: List all available search engines
- **`search_papers`**: Search for papers across engines
- **`get_paper_details`**: Get detailed information for a paper

### Project Management
- **`create_project`**: Create a new research project
- **`list_project_papers`**: List papers in a project

### Paper Management
- **`get_citing_papers`**: Find papers that cite a given paper
- **`export_paper_bibtex`**: Export paper citation in BibTeX format
- **`manage_paper_keywords`**: Add/remove keywords from papers

## Search Engines

### Built-in Engines
- **Local Database (`local`)**: Search within stored papers
- **Google Scholar (`google_scholar`)**: Academic search using SerpAPI

### Adding New Engines
1. Create a new Python file in `knowledge-base/engines/`
2. Implement a class inheriting from `BaseSearchEngine`
3. The server will automatically discover and load it

Example engine structure:
```python
from search_engines import BaseSearchEngine

class MyEngine(BaseSearchEngine):
    @property
    def name(self) -> str:
        return "My Search Engine"

    @property
    def id(self) -> str:
        return "my_engine"

    def is_available(self) -> bool:
        return True  # Check API keys, etc.

    def search(self, query: str, **kwargs):
        # Implement search logic
        return []
```

## Configuration

### Environment Variables
- `SERP_API_KEY`: Required for Google Scholar search
- `DATABASE_PATH`: SQLite database file path (default: `research.db`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `CACHE_TTL_SECONDS`: Cache expiration time (default: `3600`)

### Command Line Options
- `--directory`: Data directory (default: `./knowledge-base`)
- `--verbose, -v`: Enable verbose logging
- `--transport`: Transport method (`stdio`, `sse`, `http`)
- `--host`: Host for HTTP/SSE (default: `0.0.0.0`)
- `--port`: Port for HTTP/SSE (default: `8000`)

## Usage Examples

### Creating a Research Project
```python
# This would be called via MCP
project = await create_project("Machine Learning Research", "Study of ML algorithms")
```

### Searching for Papers
```python
# Single engine (string format)
papers = await search_papers(
    query="machine learning",
    engines="google_scholar",
    max_results=5
)

# Multiple engines (array format)
papers = await search_papers(
    query="deep learning transformers",
    engines=["google_scholar", "local"],
    max_results=10,
    project_id="ml_research"
)

# All available engines (omit engines parameter)
papers = await search_papers(
    query="artificial intelligence",
    max_results=15
)
```

### Managing Keywords
```python
# Add keywords to multiple papers
await manage_paper_keywords(
    paper_ids=["paper_1", "paper_2", "paper_3"],
    action="add",
    keywords=["machine learning", "neural networks"]
)
```

## Architecture

### Core Components
- **`main.py`**: Main server entry point and MCP tool definitions
- **`database.py`**: SQLite database management and paper storage
- **`search_engines.py`**: Search engine discovery and coordination
- **`config.py`**: Configuration management and environment variables

### Data Storage
- **SQLite Database**: Stores papers, projects, and metadata
- **Directory Structure**:
  ```
  knowledge-base/
  ├── research.db          # Main database
  ├── engines/            # Search engine modules
  ├── cache/              # API response cache
  ├── downloads/          # Downloaded files
  ├── exports/            # Export files
  └── temp/               # Temporary files
  ```

## Development

### Code Quality Standards
- Clear, human-readable code with comprehensive comments
- Proper error handling and logging
- Descriptive variable and function names
- Logical code organization

### Reference Implementation
See `sample_scholar_query.py` for the gold standard of code quality and SerpAPI integration patterns.

## License

This project is open source. See individual files for license information.