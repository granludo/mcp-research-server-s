# Research MCP Server - Project Requirements Document

## Overview

This document outlines the requirements for a comprehensive research paper management system built as an MCP (Model Context Protocol) server. The system provides unified access to multiple academic search engines and maintains a local database of research papers with advanced organization and search capabilities.

## Reference Implementation

The `sample_scholar_query.py` file serves as a comprehensive example of SerpAPI integration for Google Scholar. It demonstrates:
- Clear, human-readable code structure
- Comprehensive documentation and comments
- Proper error handling and API key management
- Rich data extraction from SerpAPI responses
- Well-structured output formatting

**Use this file as the primary reference for implementing SerpAPI-based search engines.**

The Research MCP Server is an AI-powered tool designed to facilitate scientific literature research across multiple academic platforms. It provides a standardized interface for searching, organizing, and managing academic papers from various sources while maintaining research project organization through tagging and persistent storage.

## Core Objectives

- Provide a unified interface for academic literature search across multiple platforms
- Enable organized research project management with persistent storage
- Support flexible deployment options for different integration scenarios
- Offer extensible architecture for adding new academic search sources

## Architecture Requirements

### Server Deployment Modes
The server must support multiple transport mechanisms based on launch parameters:

1. **STDIO Mode**: For local command-line usage and direct AI model integration
2. **SSE (Server-Sent Events) Mode**: For real-time web applications and streaming responses
3. **Streaming HTTP Mode**: For web service integration with continuous data streaming

### Command Line Interface
- `--verbose` flag: Enables debug output to stderr
- `--transport` parameter: Specifies deployment mode (`stdio`, `sse`, `http`)
- `--port` parameter: Port configuration for HTTP/SSE modes (default: 8000)
- `--host` parameter: Host binding for HTTP/SSE modes (default: `0.0.0.0`)
- `--directory` parameter: Directory for database and data files (default: `./knowledge-base`)

## Search Engine Integration

### Current Search Engines
1. **Local Database** (`local`)
   - Internal search within stored papers
   - Full-text search across titles, abstracts, authors, and keywords
   - Citation network analysis within stored collection
   - Manual keyword management for organization
   - No API key required
   - Always available

2. **Google Scholar Engine** (`google_scholar`)
   - SerpAPI integration
   - Requires `SERP_API_KEY` environment variable
   - Returns structured paper metadata with citations
   - Handles rate limiting and error recovery
   - Availability checked via API key validation
   - **Reference Implementation**: See `sample_scholar_query.py` for comprehensive SerpAPI usage example

3. **ArXiv Engine** (`arxiv`)
   - Direct API integration using python-arxiv library
   - Comprehensive metadata including PDF URLs, categories, and citations
   - Category-based filtering support (e.g., cs.AI, cs.SE, cs.LG)
   - Author search capabilities
   - Related paper discovery using keyword similarity
   - Trend analysis (authors, timeline, categories, keywords)
   - Export functionality (BibTeX, CSV, JSON, Markdown)
   - Natural language query processing
   - Date range filtering
   - No API key required
   - Availability checked via network connectivity

### Future Sources (Extensibility)
- ... let's leave it open

## Data Model

### Directory Structure
The `--directory` parameter specifies the root directory for all persistent data. If the directory doesn't exist, the system will create it along with all necessary subdirectories and database files at startup.

```
knowledge-base/
├── research.db          # SQLite database
├── engines/            # Search engine modules directory
├── cache/              # Cached API responses
├── downloads/          # downloaded papers
├── exports/            # Exported data files
└── temp/               # Temporary files
```

### Paper Identification
Each paper receives two unique identifiers:

#### Primary Identifier (Human-readable)
Generated from author information:
```
Format: [LastName1, FirstInitial1., LastName2, FirstInitial2., ... Year]
Example: [Alier, M., Pereira, J., Garcia-Penalvo, F. J., Casan, M. J., & Cabre, J. 2025]
```

For duplicate author sets in the same year:
- First paper: `[Author List Year]`
- Second paper: `[Author List Year-2]`
- Third paper: `[Author List Year-3]`

#### Secondary Identifier (Integer-based)
Auto-incrementing integer for efficient indexing and quick referencing:
- Starts from 1 for each new database
- Provides fast lookups and database joins
- Used internally for performance optimization

Example: Paper ID `1001` with primary identifier `[Smith, J., Johnson, A. 2024]`

### Keyword Management
Keywords are manually managed by users to enhance search capabilities and organize research materials:

#### Manual Keyword Management
- **Add Keywords**: Users can add relevant keywords to papers for better categorization
- **Remove Keywords**: Users can remove irrelevant or outdated keywords
- **Set Keywords**: Replace all existing keywords with a new set
- **Keyword Validation**: Ensures keywords are properly formatted and relevant

#### Search Integration
- **Keyword-based Search**: Search papers by manually assigned keywords
- **Full-text Keyword Search**: Keywords are indexed for fast full-text search
- **Cross-reference Search**: Keywords help discover related papers within the collection
- **Project Organization**: Keywords support project-based research organization
...

### Database Schema (SQLite3)

#### Papers Table
```sql
CREATE TABLE papers (
    paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL, -- Human-readable identifier
    project_id TEXT NOT NULL,
    title TEXT NOT NULL,
    authors TEXT NOT NULL, -- JSON array of author objects with name and profile_link
    authors_string TEXT, -- Simple comma-separated author string for display
    abstract TEXT,
    snippet TEXT, -- Brief description/snippet when abstract not available
    publication_info TEXT, -- Full publication info (journal, volume, etc.)
    publication_date TEXT,
    publication_year INTEGER,
    source TEXT NOT NULL,
    source_url TEXT,
    pdf_url TEXT,
    doi TEXT,
    citations INTEGER DEFAULT 0,
    citation_count INTEGER DEFAULT 0, -- Actual citation count from source
    cites_id TEXT, -- ID for fetching citing papers
    result_id TEXT, -- SerpAPI result ID
    citation_id TEXT, -- Author profile citation ID
    bibtex_link TEXT, -- Link to BibTeX download
    cached_page_link TEXT, -- Google cached page link
    related_pages_link TEXT, -- Related articles link
    versions_count INTEGER DEFAULT 0, -- Number of versions available
    versions_link TEXT, -- Link to all versions
    resources TEXT, -- JSON array of additional resources (PDFs, etc.)
    keywords TEXT, -- JSON array of keywords (for search and categorization)
    metadata TEXT, -- JSON object for source-specific data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Projects Table
```sql
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

## MCP Tools

### 1. `list_search_engines`
**Purpose**: Returns available search engines including modules and local database
**Parameters**: None
**Returns**: JSON array of search engine objects
```json
[
  {
    "id": "local",
    "name": "Local Database",
    "description": "Search within previously stored papers in the local database",
    "type": "internal",
    "is_available": true
  },
  {
    "id": "google_scholar",
    "name": "Google Scholar",
    "description": "Comprehensive academic search engine with citation metrics",
    "type": "external",
    "is_available": true
  },
  {
    "id": "arxiv",
    "name": "ArXiv",
    "description": "Open-access preprint repository for scientific papers",
    "type": "external",
    "is_available": true
  }
]
```

### 2. `search_papers`
**Purpose**: Search for academic papers across configured search engines (modules + local database)
**Parameters**:
- `query` (string, required): Search query
- `engines` (string or array, optional): Single engine name or list of engine IDs to search (default: all available). Include "local" to search stored papers.
- `max_results` (integer, optional): Maximum results per engine (default: 5)
- `project_id` (string, optional): Research project identifier for tagging, when searching local the results will have the new project tag added
- `date_from` (string, optional): Start date filter (YYYY-MM-DD)
- `date_to` (string, optional): End date filter (YYYY-MM-DD)

**Engine Parameter Formats**:
- Single engine: `"google_scholar"` or `"local"`
- Multiple engines: `["google_scholar", "local"]`
- All engines: `null` or omit parameter

**Usage Examples**:
```json
// Single engine (string)
{
  "query": "machine learning",
  "engines": "google_scholar",
  "max_results": 5
}

// Multiple engines (array)
{
  "query": "artificial intelligence",
  "engines": ["google_scholar", "local"],
  "max_results": 3
}

// All available engines
{
  "query": "neural networks",
  "max_results": 10
}
```

**Returns**: Array of paper objects with the following structure:
```json
[
  {
    "paper_id": 123,
    "id": "[Author, A. Year]",
    "title": "Paper Title",
    "year": 2024,
    "publication": "Journal Name, Volume(Issue), Pages",
    "authors": ["Author, A.", "Author, B."],
    "keywords": ["keyword1", "keyword2"],
    "source": "google_scholar",
    "project_id": "research_project_1"
  }
]
```

### 3. `get_paper_details`
**Purpose**: Retrieve detailed information for a specific paper
**Parameters**:
- `paper_id` (string or integer, required): Paper identifier (accepts both text and integer IDs)
**Returns**: Complete paper JSON object

### 4. `list_project_papers`
**Purpose**: List all papers associated with a research project
**Parameters**:
- `project_id` (string, required): Research project identifier
- `limit` (integer, optional): Maximum results to return
- `offset` (integer, optional): Pagination offset
**Returns**: Array of paper summaries

### 5. `create_project`
**Purpose**: Create a new research project
**Parameters**:
- `name` (string, required): Project name
- `description` (string, optional): Project description
**Returns**: Generated project ID -> we can always refer to it by name

### 6. `get_citing_papers`

**Purpose**: Retrieve papers that cite a specific paper
**Parameters**:
- `paper_id` (string or integer, required): Paper identifier (accepts both text and integer IDs)
- `max_results` (integer, optional): Maximum citing papers to retrieve (default: 10)
**Returns**: Array of citing paper information

### 7. `export_paper_bibtex`

**Purpose**: Get BibTeX citation for a paper
**Parameters**:
- `paper_id` (string or integer, required): Paper identifier (accepts both text and integer IDs)
**Returns**: BibTeX formatted citation string

### 8. `manage_paper_keywords`

**Purpose**: Add or remove keywords from one or multiple papers manually
**Parameters**:
- `paper_ids` (string, integer, or array, required): Single paper identifier or array of paper identifiers
- `action` (string, required): Action to perform ('add', 'remove', 'set')
- `keywords` (array, required): List of keywords to add/remove or set
**Returns**: Dictionary with paper_ids as keys and their updated keywords arrays as values

**Actions**:
- `'add'`: Add the specified keywords to existing ones
- `'remove'`: Remove the specified keywords from existing ones
- `'set'`: Replace all existing keywords with the specified ones

**Usage Examples**:
- Single paper: `paper_ids: "123"` or `paper_ids: 123`
- Multiple papers: `paper_ids: ["123", "456", 789]`

### 9. `search_by_author`

**Purpose**: Search for papers by a specific author across all configured engines
**Parameters**:
- `author_name` (string, required): Name of the author to search for
- `engines` (string or array, optional): Engines to search (default: all available)
- `max_results` (integer, optional): Maximum results per engine (default: 10)
- `date_from` (string, optional): Start date filter (YYYY-MM-DD)
- `date_to` (string, optional): End date filter (YYYY-MM-DD)
- `project_id` (string, optional): Research project identifier for tagging

**Returns**: Array of papers by the specified author, or error message if not supported by engine

### 10. `find_related_papers`

**Purpose**: Find papers related to a given paper using keyword similarity and citation analysis
**Parameters**:
- `paper_id` (string or integer, required): Reference paper identifier
- `engines` (string or array, optional): Engines to search (default: all available)
- `max_results` (integer, optional): Maximum related papers to return (default: 10)
- `similarity_threshold` (float, optional): Minimum similarity score (0.0 to 1.0, default: 0.7)
- `project_id` (string, optional): Research project identifier for tagging

**Returns**: Array of related papers with similarity scores, or error message if not supported

### 11. `analyze_paper_trends`

**Purpose**: Analyze trends in a collection of papers
**Parameters**:
- `paper_ids` (array, required): List of paper identifiers to analyze
- `analysis_type` (string, required): Type of analysis ('authors', 'keywords', 'timeline', 'categories')
- `project_id` (string, optional): Research project identifier

**Returns**: Analysis results based on the specified analysis type, or error message if not supported

### 12. `get_search_engine_categories`

**Purpose**: Get a list of available categories and their descriptions for supported search engines
**Parameters**:
- `engines` (string or array, optional): Engines to query (default: all available)

**Returns**: Dictionary containing categories organized by search engine and subject area

### 13. `export_search_results`

**Purpose**: Export search results to various formats
**Parameters**:
- `paper_ids` (array, required): List of paper identifiers to export
- `format` (string, required): Export format ('bibtex', 'csv', 'json', 'markdown')
- `project_id` (string, optional): Research project identifier
- `filename` (string, optional): Output filename (without extension)

**Returns**: Export result with file path and success status, or error message if not supported

## MCP Resources

### 1. `papers://{project_id}`

**Purpose**: Access papers within a specific research project
**Returns**: Markdown-formatted list of papers with metadata

### 2. `engines://list`

**Purpose**: Current status of all configured search engines (modules + local database)
**Returns**: Markdown table of search engines status and capabilities

## Error Handling

### Search Engine Failures

- Individual engine failures should not stop the entire search
- Return partial results with error information for failed engines
- Implement retry logic with exponential backoff for external engines
- Graceful handling of unavailable engines during initialization

### Database Errors

- Graceful degradation when database is unavailable
- Transaction rollback for multi-step operations
- Data validation before insertion

### Search Engine Rate Limiting

- Respect engine-specific rate limits
- Implement queuing for high-volume requests
- Cache recent results to reduce API calls
- Handle local database queries efficiently
- Dynamic rate limit adjustment based on engine availability

## Security Considerations

### API Key Management

- Secure storage of API keys via environment variables
- No logging of API keys in debug output
- Validation of API key presence before operations

### Data Privacy

- No collection of personal user data
- Research project data isolated by project ID
- Secure deletion capabilities for projects

## Performance Requirements

### Response Times

- Search operations: < 30 seconds for typical queries
- Paper retrieval: < 2 seconds for cached papers
- Project listing: < 1 second
- Citation fetching: < 10 seconds for citing papers
- BibTeX export: < 1 second for cached entries
- Keyword management: < 1 second per operation

### Scalability

- Support for concurrent searches across multiple projects
- Efficient database indexing for large paper collections
- Memory-efficient processing of large result sets
- API response caching to reduce external API calls

### Caching Strategy

- **API Response Cache**: Store SerpAPI responses in cache/ directory with TTL
- **BibTeX Cache**: Cache BibTeX citations to avoid repeated API calls
- **Citation Network Cache**: Cache citing papers data with configurable expiration
- **Keyword Cache**: Cache keyword operations for faster subsequent access
- **Search Results Cache**: Cache formatted search results for quick subsequent access
- **Database Indexing**: Optimize queries on frequently accessed fields (authors, year, citations, keywords)
- **Integer ID Optimization**: Use auto-incrementing integer IDs for fast database joins and lookups

## Search Engine Architecture

### Search Engine Modules

Each search engine is implemented as a Python module with a standardized interface:

1. **Local Database (`local`)**: Internal search within stored papers
2. **Google Scholar (`google_scholar`)**: SerpAPI integration
3. **ArXiv (`arxiv`)**: Direct API integration
4. **Future Engines**: Additional engines can be added as modules

### Search Engine Interface
Each search engine module must implement a standardized interface. **Follow the code quality standards demonstrated in `sample_scholar_query.py`** for all search engine implementations:

```python
class BaseSearchEngine:
    @property
    def name(self) -> str:
        """Human-readable name of the search engine"""
        pass

    @property
    def id(self) -> str:
        """Unique identifier for the search engine"""
        pass

    def is_available(self) -> bool:
        """Check if this search engine is available (API keys, connections)"""
        pass

    def search(self, query: str, **kwargs) -> List[dict]:
        """Search for papers and return standardized results"""
        pass

    def get_paper_details(self, paper_id: str) -> dict:
        """Retrieve detailed information for a specific paper"""
        pass

    def get_citing_papers(self, paper_id: str) -> List[dict]:
        """Get papers that cite the given paper (if supported)"""
        pass
```

### Local Database as Search Source

The local database serves as the "local" search source with the following capabilities:

- **Full-text Search**: Search within paper titles, abstracts, authors, and keywords
- **Keyword-based Search**: Find papers by manually assigned keywords
- **Metadata Filtering**: Filter by publication year, authors, journals, citations
- **Citation Network**: Find citing and cited papers within stored collection
- **Project-based Search**: Search within specific research projects
- **Advanced Queries**: Boolean search, phrase matching, and fuzzy matching

### Search Engine Discovery and Loading
- **Module Discovery**: Search engine modules are discovered from the `engines/` directory
- **Availability Check**: Each engine's `is_available()` method is called during initialization
- **Dynamic Loading**: Engines are loaded based on availability (API keys, network connectivity)
- **Error Isolation**: Engine failures don't affect other engines or core functionality
- **Runtime Updates**: Engine availability can be re-checked without restarting

## Extensibility

### Search Engine Development
- Standardized `BaseSearchEngine` interface for easy development
- Comprehensive documentation and examples
- Simple module structure for rapid prototyping
- Community-contributed search engines

### Code Quality Standards
All code must adhere to high standards of clarity and maintainability:

#### Code Clarity Requirements
- **Human-Readable**: Code should be self-explanatory and easy to understand
- **Comprehensive Comments**: Every function, class, and complex logic must have clear documentation
- **Descriptive Variable Names**: Use meaningful names that explain purpose and context
- **Logical Structure**: Organize code in clear, logical sections with appropriate separation of concerns

#### Documentation Standards
- **Function Documentation**: Include purpose, parameters, return values, and usage examples
- **Class Documentation**: Describe class purpose, responsibilities, and key methods
- **Inline Comments**: Explain complex algorithms, business logic, and non-obvious decisions
- **API Examples**: Provide practical usage examples for all public interfaces

#### Implementation Guidelines
- **Error Handling**: Implement comprehensive error handling with meaningful messages
- **Logging**: Include appropriate logging for debugging and monitoring
- **Configuration**: Make systems configurable through environment variables or config files
- **Testing**: Include unit tests and integration tests for all components
- **Reference Examples**: Use `sample_scholar_query.py` as a model for clear, well-documented SerpAPI integration

### Data Export
- JSON export of project papers
- BibTeX format generation
- CSV export for analysis tools

## Testing Requirements

### Unit Tests
- Individual tool functionality
- Database operations
- API integration mocking

### Integration Tests
- End-to-end search workflows
- Multi-source search validation
- Project management operations

## Deployment and Maintenance

### Initialization and Setup
The system automatically handles initial setup when the `--directory` parameter is specified:

1. **Directory Creation**: Creates the specified directory and all subdirectories if they don't exist
2. **Database Initialization**: Creates the SQLite database with all required tables and indexes
3. **Search Engine Discovery**: Scans the `engines/` directory and loads available search engines
4. **Availability Validation**: Tests each search engine's connectivity and API key validity
5. **Engine Registration**: Registers available engines in memory for search operations
6. **Configuration Setup**: Initializes default configuration for engines and caching
7. **Migration Handling**: Automatically handles database schema updates for existing installations

### Environment Setup
- Python 3.11+ requirement
- SQLite3 for data storage
- Required packages via requirements.txt

## Success Metrics

1. **Search Coverage**: Percentage of successful searches across all sources
2. **Response Time**: Average time for search completion
3. **Data Quality**: Accuracy of paper metadata extraction
4. **User Satisfaction**: Effectiveness in supporting research workflows


