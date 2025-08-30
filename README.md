# 🔬 Research MCP Server

*A Comprehensive Academic Research Paper Management System*

**Author**: Marc Alier
**Version**: 1.0.0
**License**: MIT

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [⚙️ Configuration](#️-configuration)
- [🔍 MCP Tools](#-mcp-tools)
- [🔧 Search Engines](#-search-engines)
- [🛡️ Deduplication System](#️-deduplication-system)
- [🤖 LLM Integration](#-llm-integration)
- [📊 Usage Examples](#-usage-examples)
- [🏗️ Architecture](#️-architecture)
- [🧪 Testing](#-testing)
- [🐛 Troubleshooting](#-troubleshooting)
- [📄 License](#-license)

---

## 🎯 Overview

The **Research MCP Server** is a sophisticated academic research management system built using the Model Context Protocol (MCP). It provides unified access to multiple academic search engines while maintaining a comprehensive local database of research papers with advanced organization, deduplication, and AI-enhanced features.

### 🎯 Mission

To create a production-ready research paper management system that combines the power of multiple academic search engines with intelligent data organization, preventing duplicates while providing seamless access to research literature.

### 🎖️ Key Differentiators

- **🛡️ Enterprise-Grade Deduplication**: Advanced multi-criteria duplicate detection
- **🤖 AI-Powered Author Extraction**: GPT-4o-mini integration for intelligent parsing
- **🔄 Unified Search Interface**: Single API for multiple academic sources
- **📊 Production-Ready**: Comprehensive error handling, logging, and optimization
- **🔧 Extensible Architecture**: Easy to add new search engines and features

---

## ✨ Key Features

### 🔍 Advanced Search & Discovery
- **Multi-Engine Search**: Google Scholar, ArXiv, Local Database
- **Unified API**: Single interface for all search engines
- **Smart Filtering**: Date ranges, publication types, citation counts
- **Real-time Results**: Parallel search across multiple engines

### 🛡️ Enterprise Deduplication System
- **Multi-Criteria Detection**: DOI, Title+Authors+Year, URL, Similarity matching
- **Smart Project Association**: Existing papers linked to new projects
- **Database Integrity**: Prevents duplicates while maintaining relationships
- **Transparent Reporting**: Clear indication of new vs existing papers

### 🤖 AI-Enhanced Processing
- **LLM Author Extraction**: GPT-4o-mini for complex author name parsing
- **4-Tier Fallback Strategy**: Structured → LLM → Regex → Empty
- **Intelligent Metadata**: Automatic field extraction and normalization
- **Graceful Degradation**: Works without API keys using regex fallbacks

### 📊 Comprehensive Management
- **Project Organization**: Group papers by research topics
- **Keyword Management**: Manual keyword assignment and bulk operations
- **Citation Tracking**: Track citing papers and citation networks
- **Export Capabilities**: BibTeX, JSON, and other formats

### 🔧 Production Features
- **Robust Error Handling**: Comprehensive exception management
- **Performance Optimization**: Database indexing and query optimization
- **Comprehensive Logging**: Debug and production logging levels
- **Environment Configuration**: Flexible deployment options

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- SQLite 3
- Internet connection for API access
- Optional: OpenAI API key for enhanced features

### Minimal Setup (3 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/granludo/mcp-research-server-s.git
cd mcp-research-server-s

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up minimal configuration
echo "SERP_API_KEY=your_serpapi_key_here" > .env

# 4. Run the server
python main.py --verbose

# 5. Test basic functionality
curl -X POST http://localhost:8000/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "list_search_engines"}'
```

### 🎉 Success Indicators
- Server starts without errors
- At least 2 search engines are available (Local Database + Google Scholar/ArXiv)
- Database tables are created automatically
- All MCP tools are registered successfully

---

## 📦 Installation

### Standard Installation

   ```bash
# Clone the repository
git clone https://github.com/granludo/mcp-research-server-s.git
cd mcp-research-server-s

# Install Python dependencies
pip install -r requirements.txt

# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Advanced Installation

#### With OpenAI Integration (Recommended)
```bash
# Install with LLM support
pip install -r requirements.txt

# Add OpenAI API key for enhanced author extraction
echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
```

#### Docker Installation
```bash
# Build Docker image
docker build -t research-mcp-server .

# Run with environment variables
docker run -p 8000:8000 \
  -e SERP_API_KEY=your_key \
  -e OPENAI_API_KEY=your_key \
  research-mcp-server
```

### Required Dependencies

#### Core Dependencies
- `fastmcp>=0.9.0` - MCP server framework
- `python-dotenv>=1.0.0` - Environment configuration
- `requests>=2.28.0` - HTTP client for APIs
- `httpx>=0.24.0` - Async HTTP client

#### Search Engine Dependencies
- `arxiv>=1.4.0` - ArXiv API integration
- `scikit-learn>=1.3.0` - Advanced NLP features

#### AI Enhancement (Optional)
- `openai>=1.0.0` - GPT-4o-mini for author extraction

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required for Google Scholar search
SERP_API_KEY=your_serpapi_key_here

# Optional: Enhanced AI features
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Additional search engines
SEMANTIC_SCHOLAR_API_KEY=your_key_here
PUBMED_API_KEY=your_key_here

# Optional: Customization
DATABASE_PATH=./knowledge-base/research.db
LOG_LEVEL=INFO
CACHE_TTL_SECONDS=3600
```

### API Keys Setup

#### SerpAPI (Google Scholar)
1. Sign up at [SerpAPI](https://serpapi.com/)
2. Get your API key from the dashboard
3. Add to `.env`: `SERP_API_KEY=your_key`

#### OpenAI (Enhanced Features)
1. Sign up at [OpenAI](https://platform.openai.com/)
2. Create an API key
3. Add to `.env`: `OPENAI_API_KEY=your_key`
4. **Note**: System works without this, but with reduced author extraction accuracy

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --directory PATH          Data directory (default: ./knowledge-base)
  --verbose, -v            Enable verbose logging
  --transport TEXT         Transport method: stdio, sse, http (default: stdio)
  --host TEXT             Host for HTTP/SSE (default: 0.0.0.0)
  --port INTEGER          Port for HTTP/SSE (default: 8000)
  --help                  Show help message
```

### Transport Methods

#### STDIO (Default)
```bash
python main.py
# Best for MCP clients and AI assistants
```

#### HTTP Server
```bash
python main.py --transport http --port 8000
# RESTful API access
```

#### Server-Sent Events
```bash
python main.py --transport sse --port 8000
# Real-time streaming responses
```

---

## 🔍 MCP Tools

### Core Search Tools

#### `list_search_engines`
Lists all available search engines with their status.

**Parameters**: None

**Response**:
```json
{
  "engines": [
    {
      "id": "google_scholar",
      "name": "Google Scholar",
      "status": "available",
      "description": "Academic search using SerpAPI"
    },
    {
      "id": "arxiv",
      "name": "ArXiv",
      "status": "available",
      "description": "Open access preprint repository"
    },
    {
      "id": "local",
      "name": "Local Database",
      "status": "available",
      "description": "Search within stored papers"
    }
  ]
}
```

#### `search_papers`
Search for academic papers across multiple engines.

**Parameters**:
- `query` (string, required): Search query
- `engines` (string/array, optional): Engine IDs to search
- `max_results` (integer, optional): Maximum results per engine (default: 5)
- `project_id` (string, optional): Associate results with project
- `date_from` (string, optional): Start date (YYYY-MM-DD)
- `date_to` (string, optional): End date (YYYY-MM-DD)

**Response**:
```json
{
  "query": "machine learning",
  "results": [...],
  "total_count": 15,
  "project_id": "ml_research",
  "deduplication_info": {
    "new_papers": 12,
    "existing_papers": 3,
    "total_processed": 15
  }
}
```

### Advanced Research Tools

#### `search_by_author`
Search for papers by specific authors.

#### `find_related_papers`
Find papers related to a given paper.

#### `analyze_paper_trends`
Analyze publication trends and author patterns.

---

## 🔧 Search Engines

### Built-in Engines

#### Google Scholar (`google_scholar`)
- **API**: SerpAPI integration
- **Coverage**: Comprehensive academic literature
- **Features**: Citation counts, publication info, PDF links
- **Requirements**: `SERP_API_KEY` environment variable

#### ArXiv (`arxiv`)
- **API**: Native ArXiv API
- **Coverage**: Physics, Mathematics, Computer Science, etc.
- **Features**: Categories, submission dates, PDF downloads
- **Requirements**: `arxiv` Python package

#### Local Database (`local`)
- **API**: SQLite database queries
- **Coverage**: Previously stored papers
- **Features**: Fast local search, project filtering
- **Requirements**: None (always available)

### Adding New Engines

1. **Create Engine File**:
   ```python
   # engines/my_engine.py
   from search_engines import BaseSearchEngine

   class MyEngine(BaseSearchEngine):
       @property
       def name(self) -> str:
           return "My Academic Search Engine"

       @property
       def id(self) -> str:
           return "my_engine"

       def is_available(self) -> bool:
           # Check API keys, connectivity, etc.
           return True

       def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
           # Implement search logic
           # Return standardized paper dictionaries
           return papers
   ```

2. **Engine Discovery**:
   - Place in `engines/` directory
   - Server automatically discovers and loads
   - Appears in `list_search_engines` output

---

## 🛡️ Deduplication System

The Research MCP Server features an enterprise-grade deduplication system that prevents duplicate papers while maintaining data integrity.

### Deduplication Criteria (Priority Order)

#### 1. DOI Match (Highest Priority)
- Exact DOI comparison
- Most reliable unique identifier
- Used for identical papers across sources

#### 2. Title + Authors + Year Match
- Normalized title comparison
- Author list matching
- Publication year verification
- Handles slight title variations

#### 3. Source URL Match
- Exact URL comparison
- Useful for same source, different metadata

#### 4. Title Similarity Match (Fallback)
- Word overlap analysis (>70% similarity)
- Catches papers with different titles but same content

### How It Works

```python
# When storing search results
stored_papers = db_manager.store_search_results(results, project_id)

# Internal deduplication logic:
# 1. Check each paper against existing database
# 2. If duplicate found: associate with project, return existing
# 3. If new paper: insert into database, assign new ID
# 4. Return all papers with proper database IDs
```

### Smart Project Association

```python
# For existing papers found during deduplication
if existing_paper:
    # Associate paper with new project (if not already)
    UPDATE papers SET project_id = ? WHERE paper_id = ?

    # Return existing paper with proper metadata
    existing_paper['is_new_paper'] = False
    return existing_paper
```

### Transparency and Reporting

**API Response includes deduplication info**:
```json
{
  "results": [...],
  "deduplication_info": {
    "new_papers": 8,
    "existing_papers": 3,
    "total_processed": 11
  }
}
```

---

## 🤖 LLM Integration

The Research MCP Server integrates GPT-4o-mini for intelligent author extraction when structured data is unavailable.

### 4-Tier Fallback Strategy

#### 1. Structured Data (Fastest)
```python
# Direct from API response
authors = pub_info.get("authors", [])
if authors:
    return authors  # Use structured data
```

#### 2. LLM-Enhanced Parsing (Smart)
```python
# When structured data missing
if not authors_structured:
    authors = llm_extract_authors(summary)
    if authors:
        return authors  # Use LLM extraction
```

#### 3. Regex-Based Parsing (Reliable)
```python
# When LLM unavailable or fails
authors = regex_extract_authors(summary)
if authors:
    return authors  # Use regex extraction
```

#### 4. Graceful Fallback (Always Works)
```python
# Final fallback
return []  # Empty list, but system continues
```

### LLM Prompt Engineering

**Intelligent Author Extraction**:
```
Extract author names from this academic paper citation. Return only the author names as a JSON array.
Keep the names in the same format as they appear in the citation.

Citation: "T.C. Ma, David E. Willis - Frontiers in Neuroscience, 2015"
Output: ["T.C. Ma", "David E. Willis"]
```

### Graceful Degradation

**Without OpenAI API Key**:
- System works normally
- Falls back to regex parsing
- No functionality loss
- Clear logging of unavailable features

**With OpenAI API Key**:
- Enhanced author extraction accuracy
- Handles complex name formats
- Better edge case handling
- Improved overall data quality

---

## 📊 Usage Examples

### Basic Search Operations

#### Search Across Multiple Engines
```python
# Search with automatic deduplication
papers = await search_papers(
    query="machine learning transformers",
    engines=["google_scholar", "arxiv"],
    max_results=10,
    project_id="nlp_research"
)

print(f"Found {papers['total_count']} papers")
print(f"New papers: {papers['deduplication_info']['new_papers']}")
print(f"Existing papers: {papers['deduplication_info']['existing_papers']}")
```

#### Create and Manage Projects
```python
# Create research project
project = await create_project(
    name="Deep Learning Research",
    description="Study of neural networks and AI"
)
project_id = project['project_id']

# Search and associate with project
papers = await search_papers(
    query="neural networks",
    project_id=project_id,
    max_results=20
)
```

---

## 🏗️ Architecture

### Core Components

```
Research MCP Server
├── main.py                 # MCP server, tool definitions, API endpoints
├── search_engines.py       # Engine discovery, LLM integration, parsing
├── database.py            # SQLite management, deduplication, queries
├── config.py              # Environment configuration, API keys
└── engines/               # Search engine implementations
    ├── google_scholar.py  # SerpAPI integration
    ├── arxiv_engine.py    # ArXiv API integration
    └── __init__.py        # Package initialization
```

### Data Flow Architecture

```
User Query → ResearchMCPServer.search_papers()
    ↓
Parallel Engine Search
    ├─ Google Scholar API → SerpAPI
    ├─ ArXiv API → Native
    └─ Local Database → SQLite
    ↓
Results Aggregation
    ↓
DatabaseManager.store_search_results() ← DEDUPLICATION HAPPENS HERE
    ↓
Deduplicated Results with Database IDs
    ↓
Response to User with Proper IDs
```

---

## 🧪 Testing

### Comprehensive Test Suite

The project includes a comprehensive test suite covering all major functionality:

```bash
# Run all tests
python comprehensive_test_suite.py

# Run specific test categories
python -c "from comprehensive_test_suite import ComprehensiveTestSuite; import asyncio; suite = ComprehensiveTestSuite(); asyncio.run(suite.test_deduplication())"
```

### Test Categories

- **Server Deployment & CLI**: Directory creation, transport methods
- **Search Engine Integration**: Engine discovery, availability, search functionality
- **Data Model & Database**: Schema validation, paper identification, project management
- **MCP Tools**: All tool functionality and parameter validation
- **Error Handling**: Engine failures, API key validation
- **Performance**: Response times, concurrent operations
- **Configuration & Logging**: Environment variables, logging configuration
- **LLM Integration**: Author extraction with AI enhancement

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Search Engines Not Available
**Problem**: `list_search_engines` shows engines as unavailable

**Solutions**:
```bash
# Check API keys
echo $SERP_API_KEY
echo $OPENAI_API_KEY

# Verify environment file
cat .env

# Test API connectivity
python -c "import requests; print(requests.get('https://serpapi.com/search.json?engine=google_scholar&q=test&api_key=$SERP_API_KEY').status_code)"
```

#### 2. LLM Features Not Working
**Problem**: Author extraction falling back to regex

**Solutions**:
```bash
# Check OpenAI API key
echo $OPENAI_API_KEY

# Test OpenAI connectivity
python -c "
import openai
client = openai.OpenAI(api_key='$OPENAI_API_KEY')
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'Hello'}]
)
print('OpenAI working')
"
```

#### 3. Database Connection Issues
**Problem**: "Unable to connect to database"

**Solutions**:
```bash
# Check file permissions
ls -la knowledge-base/research.db

# Reset database
rm knowledge-base/research.db
python main.py  # Recreates database
```

---

## 📄 License

**MIT License**

Copyright (c) 2024 Marc Alier

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 🙏 Acknowledgments

- **OpenAI** for GPT-4o-mini API enabling intelligent author extraction
- **SerpAPI** for reliable Google Scholar search integration
- **ArXiv** for open access to scientific literature
- **FastMCP** for the robust MCP server framework
- **Python Community** for excellent libraries and tools

---

## 📞 Contact

**Marc Alier**
- **Email**: marc.alier@example.com (placeholder)
- **GitHub**: [@granludo](https://github.com/granludo)
- **LinkedIn**: [Marc Alier](https://linkedin.com/in/marc-alier) (placeholder)

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**

*Built with ❤️ by Marc Alier for the academic research community*

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