#!/usr/bin/env python3
"""
Research MCP Server - Main Entry Point

A comprehensive research paper management system built as an MCP server.
Provides unified access to multiple academic search engines and maintains
a local database of research papers with advanced organization capabilities.

Follows the code quality standards established in sample_scholar_query.py:
- Clear, human-readable code structure
- Comprehensive documentation and comments
- Proper error handling and logging
- Well-organized modules and functions
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

from mcp.server.fastmcp import FastMCP

# Import our custom modules
from database import DatabaseManager
from search_engines import SearchEngineManager
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),  # Use stderr for MCP compatibility
        logging.FileHandler('knowledge-base/server.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.debug("✅ Loaded environment variables from .env file")
except ImportError:
    logger.warning("⚠️  python-dotenv not installed, .env file will not be loaded")
except Exception as e:
    logger.warning(f"⚠️  Error loading .env file: {e}")


class ResearchMCPServer:
    """
    Main Research MCP Server class.

    This class coordinates all the components of the research server:
    - Database management for papers and projects
    - Search engine discovery and management
    - MCP tool registration and handling
    - Configuration management
    """

    def __init__(self, data_directory: str = "./knowledge-base", verbose: bool = False, transport: str = "stdio"):
        """
        Initialize the Research MCP Server.

        Args:
            data_directory: Directory for database and data files
            verbose: Enable verbose logging
            transport: Transport mode (stdio, sse, http)
        """
        self.data_directory = Path(data_directory)
        self.verbose = verbose
        self.transport = transport

        # Set up logging level
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("Verbose logging enabled")

        logger.info(f"Initializing Research MCP Server with data directory: {data_directory}")

        # Log environment information for debugging
        if verbose:
            logger.debug("Environment debugging information:")
            logger.debug(f"Current working directory: {os.getcwd()}")
            logger.debug(f"SERP_API_KEY environment variable: {'***' + os.getenv('SERP_API_KEY', '')[-4:] if os.getenv('SERP_API_KEY') else 'Not set'}")
            logger.debug(f"All environment variables with 'SERP': {[k for k in os.environ.keys() if 'SERP' in k.upper()]}")
            logger.debug(f"All environment variables with 'API': {[k for k in os.environ.keys() if 'API' in k.upper()]}")
            logger.debug(f"Python path entries: {sys.path[:3]}...")

        # Initialize configuration
        self.config = Config()

        # Initialize database manager
        self.db_manager = DatabaseManager(self.data_directory)

        # Initialize search engine manager
        # Engines are now in the root-level engines/ directory, not in knowledge-base/
        engines_dir = Path(__file__).parent / "engines"
        self.search_engine_manager = SearchEngineManager(engines_dir, self.db_manager)

        # Initialize MCP server
        self.mcp_server = FastMCP("research-scholar")

    @property
    def engine_manager(self):
        """Alias for search_engine_manager for backward compatibility."""
        return self.search_engine_manager

    def _register_tools(self):
        """Register all MCP tools with the server."""
        logger.debug("Registering MCP tools...")

        # Core search and listing tools
        self.mcp_server.tool()(self.list_search_engines)
        self.mcp_server.tool()(self.search_papers)
        self.mcp_server.tool()(self.get_paper_details)

        # Project management tools
        self.mcp_server.tool()(self.create_project)
        self.mcp_server.tool()(self.list_project_papers)

        # Paper management tools
        self.mcp_server.tool()(self.get_citing_papers)
        self.mcp_server.tool()(self.export_paper_bibtex)
        self.mcp_server.tool()(self.manage_paper_keywords)

        # Advanced search and analysis tools
        self.mcp_server.tool()(self.search_by_author)
        self.mcp_server.tool()(self.find_related_papers)
        self.mcp_server.tool()(self.analyze_paper_trends)
        self.mcp_server.tool()(self.get_search_engine_categories)
        self.mcp_server.tool()(self.export_search_results)

        logger.debug("MCP tools registration complete")

    def _register_resources(self):
        """Register all MCP resources with the server."""
        logger.debug("Registering MCP resources...")

        # Register project papers resource
        self.mcp_server.resource(
            "papers://{project_id}",
            name="Project Papers",
            description="Access papers within a specific research project",
            mime_type="text/markdown"
        )(self.get_project_papers_resource)

        # Register search engines status resource
        self.mcp_server.resource(
            "engines://list",
            name="Search Engines Status",
            description="Current status of all configured search engines",
            mime_type="text/markdown"
        )(self.get_search_engines_resource)

        logger.debug("MCP resources registration complete")

    async def initialize(self):
        """
        Initialize all server components.

        This method:
        1. Creates necessary directories
        2. Initializes the database
        3. Discovers and loads search engines
        4. Sets up caching systems
        """
        logger.info("Starting server initialization...")

        try:
            # Create necessary directories
            await self._create_directories()

            # Initialize database
            await self.db_manager.initialize()

            # Discover and load search engines
            await self.search_engine_manager.discover_engines()

            # Register MCP tools and resources
            self._register_tools()
            self._register_resources()

            logger.info("Server initialization completed successfully")

        except Exception as e:
            logger.error(f"Server initialization failed: {e}")
            raise

    async def _create_directories(self):
        """Create all necessary directories for the server."""
        directories = [
            self.data_directory,
            self.data_directory / "engines",
            self.data_directory / "cache",
            self.data_directory / "downloads",
            self.data_directory / "exports",
            self.data_directory / "temp"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")

    # MCP Tool Implementations

    def list_search_engines(self) -> Dict:
        """
        Returns all discovered search engines including modules and local database.

        Returns:
            Dictionary containing all search engines with their status
        """
        try:
            engines = self.search_engine_manager.get_all_discovered_engines()

            result = {
                "engines": [
                    {
                        "id": engine.id,
                        "name": engine.name,
                        "description": getattr(engine, 'description', ''),
                        "type": "external" if engine.id != "local" else "internal",
                        "is_available": engine.is_available()
                    }
                    for engine in engines
                ],
                "total_count": len(engines)
            }

            logger.debug(f"Listed {len(engines)} search engines")
            return result

        except Exception as e:
            logger.error(f"Error listing search engines: {e}")
            return {"error": str(e), "engines": [], "total_count": 0}

    def search_papers(
        self,
        query: str,
        engines: Optional[Union[str, List[str]]] = None,
        max_results: int = 5,
        project_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict:
        """
        Search for academic papers across configured search engines.

        Args:
            query: Search query
            engines: Single engine name (string) or list of engine IDs to search (default: all available)
            max_results: Maximum results per engine
            project_id: Research project identifier for tagging
            date_from: Start date filter (YYYY-MM-DD)
            date_to: End date filter (YYYY-MM-DD)

        Returns:
            Dictionary containing search results
        """
        try:
            logger.info(f"Searching for papers with query: '{query}'")

            # Normalize engines parameter to list format
            if isinstance(engines, str):
                # Single engine name as string
                normalized_engines = [engines]
                logger.debug(f"Converted single engine '{engines}' to list: {normalized_engines}")
            elif isinstance(engines, list):
                # Already a list
                normalized_engines = engines
            else:
                # None or other type, use all available
                normalized_engines = None

            # Perform the search
            results = self.search_engine_manager.search_papers(
                query=query,
                engines=normalized_engines,
                max_results=max_results,
                project_id=project_id,
                date_from=date_from,
                date_to=date_to
            )

            # Set project_id on results if provided
            if project_id:
                for result in results:
                    result['project_id'] = project_id

                # Store results in database with deduplication
                stored_papers = self.db_manager.store_search_results(results, project_id)
                logger.info(f"Processed {len(stored_papers)} papers for project '{project_id}'")

                # Use stored papers (which include database IDs) in response
                response_results = stored_papers

                # Add metadata about new vs existing papers
                new_papers = sum(1 for paper in stored_papers if paper.get('is_new_paper', False))
                existing_papers = len(stored_papers) - new_papers
            else:
                response_results = results
                new_papers = 0
                existing_papers = 0

            response = {
                "query": query,
                "results": response_results,
                "total_count": len(response_results),
                "project_id": project_id
            }

            # Add deduplication info if papers were stored
            if project_id:
                response["deduplication_info"] = {
                    "new_papers": new_papers,
                    "existing_papers": existing_papers,
                    "total_processed": len(response_results)
                }

            return response

        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return {"error": str(e), "results": [], "total_count": 0}

    def get_paper_details(self, paper_id: Union[str, int]) -> Dict:
        """
        Retrieve detailed information for a specific paper.

        Args:
            paper_id: Paper identifier (string or integer)

        Returns:
            Dictionary containing complete paper information
        """
        try:
            logger.debug(f"Retrieving details for paper: {paper_id}")

            paper_details = self.db_manager.get_paper_details(paper_id)

            if paper_details:
                return paper_details
            else:
                return {"error": f"Paper with ID '{paper_id}' not found"}

        except Exception as e:
            logger.error(f"Error retrieving paper details: {e}")
            return {"error": str(e)}

    def create_project(self, name: str, description: Optional[str] = None) -> Dict:
        """
        Create a new research project.

        Args:
            name: Project name
            description: Optional project description

        Returns:
            Dictionary containing project information
        """
        try:
            logger.info(f"Creating new project: {name}")

            project = self.db_manager.create_project(name, description)

            return {
                "project_id": project["project_id"],
                "name": project["name"],
                "description": project["description"],
                "created_at": project["created_at"]
            }

        except Exception as e:
            logger.error(f"Error creating project: {e}")
            return {"error": str(e)}

    def list_project_papers(self, project_id: str, limit: int = 50, offset: int = 0) -> Dict:
        """
        List all papers associated with a research project.

        Args:
            project_id: Research project identifier
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            Dictionary containing project papers
        """
        try:
            logger.debug(f"Listing papers for project: {project_id}")

            papers = self.db_manager.get_project_papers(project_id, limit, offset)

            return {
                "project_id": project_id,
                "papers": papers,
                "count": len(papers),
                "limit": limit,
                "offset": offset
            }

        except Exception as e:
            logger.error(f"Error listing project papers: {e}")
            return {"error": str(e), "papers": [], "count": 0}

    def get_citing_papers(self, paper_id: Union[str, int], max_results: int = 10) -> Dict:
        """
        Retrieve papers that cite a specific paper.

        Args:
            paper_id: Paper identifier
            max_results: Maximum citing papers to retrieve

        Returns:
            Dictionary containing citing papers
        """
        try:
            logger.debug(f"Finding citing papers for: {paper_id}")

            citing_papers = self.db_manager.get_citing_papers(paper_id, max_results)

            return {
                "paper_id": paper_id,
                "citing_papers": citing_papers,
                "count": len(citing_papers)
            }

        except Exception as e:
            logger.error(f"Error retrieving citing papers: {e}")
            return {"error": str(e), "citing_papers": [], "count": 0}

    def export_paper_bibtex(self, paper_id: Union[str, int]) -> str:
        """
        Get BibTeX citation for a paper.

        Args:
            paper_id: Paper identifier

        Returns:
            BibTeX formatted citation string
        """
        try:
            logger.debug(f"Exporting BibTeX for paper: {paper_id}")

            bibtex = self.db_manager.get_paper_bibtex(paper_id)

            if bibtex:
                return bibtex
            else:
                return f"% Error: BibTeX not available for paper {paper_id}"

        except Exception as e:
            logger.error(f"Error exporting BibTeX: {e}")
            return f"% Error: {str(e)}"

    def manage_paper_keywords(
        self,
        paper_ids: Union[str, int, List[Union[str, int]]],
        action: str,
        keywords: List[str]
    ) -> Dict:
        """
        Add or remove keywords from one or multiple papers manually.

        Args:
            paper_ids: Single paper identifier or array of paper identifiers
            action: Action to perform ('add', 'remove', 'set')
            keywords: List of keywords to add/remove or set

        Returns:
            Dictionary with paper_ids as keys and their updated keywords arrays as values
        """
        try:
            logger.debug(f"Managing keywords for papers: {paper_ids}, action: {action}")

            # Ensure paper_ids is a list
            if not isinstance(paper_ids, list):
                paper_ids = [paper_ids]

            results = {}
            for paper_id in paper_ids:
                try:
                    updated_keywords = self.db_manager.manage_paper_keywords(
                        paper_id, action, keywords
                    )
                    results[str(paper_id)] = updated_keywords
                except Exception as e:
                    results[str(paper_id)] = {"error": str(e)}

            return {
                "action": action,
                "keywords": keywords,
                "results": results
            }

        except Exception as e:
            logger.error(f"Error managing paper keywords: {e}")
            return {"error": str(e), "results": {}}

    def search_by_author(
        self,
        author_name: str,
        engines: Optional[Union[str, List[str]]] = None,
        max_results: int = 10,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Dict:
        """
        Search for papers by a specific author across all configured engines.

        Args:
            author_name: Name of the author to search for
            engines: Engines to search (default: all available)
            max_results: Maximum results per engine
            date_from: Start date filter (YYYY-MM-DD)
            date_to: End date filter (YYYY-MM-DD)
            project_id: Research project identifier for tagging

        Returns:
            Dictionary containing search results
        """
        try:
            logger.info(f"Searching for papers by author: {author_name}")

            # Normalize engines parameter
            if isinstance(engines, str):
                normalized_engines = [engines]
            elif isinstance(engines, list):
                normalized_engines = engines
            else:
                normalized_engines = None

            all_results = []

            # Get available engines
            if normalized_engines is None:
                search_engines = self.engine_manager.get_available_engines()
            else:
                search_engines = []
                for engine_id in normalized_engines:
                    engine = self.engine_manager.get_engine(engine_id)
                    if engine and engine.is_available():
                        search_engines.append(engine)

            # Search each engine
            for engine in search_engines:
                try:
                    logger.debug(f"Searching {engine.name} for author: {author_name}")
                    results = engine.search_by_author(
                        author_name=author_name,
                        max_results=max_results,
                        date_from=date_from,
                        date_to=date_to,
                        project_id=project_id
                    )

                    # Filter out error messages and add engine info
                    valid_results = []
                    for result in results:
                        if not isinstance(result, dict) or "error" not in result:
                            result["source"] = engine.id
                            valid_results.append(result)

                    all_results.extend(valid_results)

                except Exception as e:
                    logger.warning(f"Error searching {engine.name}: {e}")

            # Store results in database if project specified
            if project_id and all_results:
                stored_papers = self.db_manager.store_search_results(all_results, project_id)
                logger.info(f"Processed {len(stored_papers)} papers for project '{project_id}'")

                # Use stored papers (which include database IDs) in response
                response_results = stored_papers[:max_results]
            else:
                response_results = all_results[:max_results]

            return {
                "query": f"author:{author_name}",
                "total_count": len(response_results),
                "results": response_results,
                "engines_searched": len(search_engines),
                "project_id": project_id
            }

        except Exception as e:
            logger.error(f"Error in author search: {e}")
            return {"error": str(e), "results": [], "total_count": 0}

    def find_related_papers(
        self,
        paper_id: Union[str, int],
        engines: Optional[Union[str, List[str]]] = None,
        max_results: int = 10,
        similarity_threshold: float = 0.7,
        project_id: Optional[str] = None
    ) -> Dict:
        """
        Find papers related to a given paper using keyword similarity and citation analysis.

        Args:
            paper_id: Reference paper identifier
            engines: Engines to search (default: all available)
            max_results: Maximum related papers to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            project_id: Research project identifier for tagging

        Returns:
            Dictionary containing related papers
        """
        try:
            logger.info(f"Finding papers related to: {paper_id}")

            # Get reference paper details
            reference_paper = self.db_manager.get_paper_details(paper_id)
            if not reference_paper:
                return {"error": f"Reference paper {paper_id} not found", "results": []}

            # Normalize engines parameter
            if isinstance(engines, str):
                normalized_engines = [engines]
            elif isinstance(engines, list):
                normalized_engines = engines
            else:
                normalized_engines = None

            all_results = []

            # Get available engines
            if normalized_engines is None:
                search_engines = self.engine_manager.get_available_engines()
            else:
                search_engines = []
                for engine_id in normalized_engines:
                    engine = self.engine_manager.get_engine(engine_id)
                    if engine and engine.is_available():
                        search_engines.append(engine)

            # Search each engine for related papers
            for engine in search_engines:
                try:
                    logger.debug(f"Finding related papers in {engine.name}")
                    results = engine.find_related_papers(
                        reference_paper=reference_paper,
                        max_results=max_results,
                        similarity_threshold=similarity_threshold,
                        project_id=project_id
                    )

                    # Filter out error messages and add engine info
                    valid_results = []
                    for result in results:
                        if not isinstance(result, dict) or "error" not in result:
                            result["source"] = engine.id
                            valid_results.append(result)

                    all_results.extend(valid_results)

                except Exception as e:
                    logger.warning(f"Error finding related papers in {engine.name}: {e}")

            # Sort by similarity score and limit results
            all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            all_results = all_results[:max_results]

            # Store results in database if project specified
            if project_id and all_results:
                stored_papers = self.db_manager.store_search_results(all_results, project_id)
                logger.info(f"Processed {len(stored_papers)} related papers for project '{project_id}'")

                # Use stored papers (which include database IDs) in response
                response_results = stored_papers
                total_count = len(stored_papers)
            else:
                response_results = all_results
                total_count = len(all_results)

            return {
                "reference_paper_id": paper_id,
                "reference_title": reference_paper.get("title", "Unknown"),
                "total_count": total_count,
                "results": response_results,
                "engines_searched": len(search_engines),
                "similarity_threshold": similarity_threshold,
                "project_id": project_id
            }

        except Exception as e:
            logger.error(f"Error finding related papers: {e}")
            return {"error": str(e), "results": [], "total_count": 0}

    def analyze_paper_trends(
        self,
        paper_ids: List[Union[str, int]],
        analysis_type: str,
        project_id: Optional[str] = None
    ) -> Dict:
        """
        Analyze trends in a collection of papers.

        Args:
            paper_ids: List of paper identifiers to analyze
            analysis_type: Type of analysis ('authors', 'keywords', 'timeline', 'categories')
            project_id: Research project identifier

        Returns:
            Analysis results based on the specified analysis type
        """
        try:
            logger.info(f"Analyzing {len(paper_ids)} papers with type: {analysis_type}")

            # Get paper details for analysis
            papers = []
            for paper_id in paper_ids:
                paper = self.db_manager.get_paper_details(paper_id)
                if paper:
                    papers.append(paper)

            if not papers:
                return {"error": "No valid papers found for analysis", "total_papers": 0}

            # Use local engine for analysis (most comprehensive)
            local_engine = self.engine_manager.get_engine("local")
            if not local_engine:
                return {"error": "Local engine not available for analysis"}

            # Perform analysis
            analysis_result = local_engine.analyze_paper_trends(papers, analysis_type)

            return {
                "analysis_type": analysis_type,
                "total_papers_analyzed": len(papers),
                "project_id": project_id,
                **analysis_result
            }

        except Exception as e:
            logger.error(f"Error analyzing paper trends: {e}")
            return {"error": str(e), "total_papers_analyzed": 0}

    def get_search_engine_categories(
        self,
        engines: Optional[Union[str, List[str]]] = None
    ) -> Dict:
        """
        Get a list of available categories and their descriptions for supported search engines.

        Args:
            engines: Engines to query (default: all available)

        Returns:
            Dictionary containing categories organized by search engine
        """
        try:
            logger.info("Getting search engine categories")

            # Normalize engines parameter
            if isinstance(engines, str):
                normalized_engines = [engines]
            elif isinstance(engines, list):
                normalized_engines = engines
            else:
                normalized_engines = None

            # Get available engines
            if normalized_engines is None:
                search_engines = self.engine_manager.get_available_engines()
            else:
                search_engines = []
                for engine_id in normalized_engines:
                    engine = self.engine_manager.get_engine(engine_id)
                    if engine and engine.is_available():
                        search_engines.append(engine)

            all_categories = {}

            # Get categories from each engine
            for engine in search_engines:
                try:
                    categories = engine.get_search_engine_categories()
                    if categories:
                        all_categories[engine.id] = categories
                except Exception as e:
                    logger.warning(f"Error getting categories from {engine.name}: {e}")

            return {
                "total_engines": len(search_engines),
                "engines_with_categories": len(all_categories),
                "categories": all_categories
            }

        except Exception as e:
            logger.error(f"Error getting search engine categories: {e}")
            return {"error": str(e), "total_engines": 0, "categories": {}}

    def export_search_results(
        self,
        paper_ids: List[Union[str, int]],
        format: str,
        project_id: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Dict:
        """
        Export search results to various formats.

        Args:
            paper_ids: List of paper identifiers to export
            format: Export format ('bibtex', 'csv', 'json', 'markdown')
            project_id: Research project identifier
            filename: Output filename (without extension)

        Returns:
            Export result with file path and success status
        """
        try:
            logger.info(f"Exporting {len(paper_ids)} papers in {format} format")

            # Get paper details for export
            papers = []
            for paper_id in paper_ids:
                paper = self.db_manager.get_paper_details(paper_id)
                if paper:
                    papers.append(paper)

            if not papers:
                return {"error": "No valid papers found for export", "papers_exported": 0}

            # Use local engine for export (most comprehensive)
            local_engine = self.engine_manager.get_engine("local")
            if not local_engine:
                return {"error": "Local engine not available for export"}

            # Perform export
            export_result = local_engine.export_search_results(
                papers=papers,
                format_type=format,
                filename=filename
            )

            return {
                "format": format,
                "papers_requested": len(paper_ids),
                "papers_exported": len(papers),
                "project_id": project_id,
                **export_result
            }

        except Exception as e:
            logger.error(f"Error exporting search results: {e}")
            return {"error": str(e), "papers_exported": 0}

    # MCP Resource Implementations

    def get_project_papers_resource(self, project_id: str) -> str:
        """
        MCP Resource: papers://{project_id}

        Access papers within a specific research project.
        Returns a markdown-formatted list of papers with metadata.

        Args:
            project_id: Research project identifier

        Returns:
            Markdown-formatted string containing project papers
        """
        try:
            logger.info(f"Accessing papers resource for project: {project_id}")

            # Get papers for the project
            papers = self.db_manager.get_project_papers(project_id)

            if not papers:
                return f"# Papers in Project '{project_id}'\n\nNo papers found in this project."

            # Format as markdown
            markdown = f"# Papers in Project '{project_id}'\n\n"
            markdown += f"**Total Papers:** {len(papers)}\n\n"

            for i, paper in enumerate(papers, 1):
                paper_id = paper.get('paper_id', paper.get('id', 'Unknown'))
                title = paper.get('title', 'Unknown Title')
                authors = paper.get('authors', 'Unknown Authors')
                year = paper.get('year', paper.get('publication_year', 'Unknown Year'))
                publication = paper.get('publication', paper.get('publication_info', 'Unknown Publication'))

                markdown += f"## {i}. {title}\n\n"
                markdown += f"**Authors:** {authors}\n\n"
                markdown += f"**Year:** {year}\n\n"
                markdown += f"**Publication:** {publication}\n\n"
                markdown += f"**Paper ID:** {paper_id}\n\n"
                markdown += "---\n\n"

            return markdown

        except Exception as e:
            logger.error(f"Error accessing project papers resource: {e}")
            return f"# Error Accessing Project Papers\n\nFailed to retrieve papers for project '{project_id}': {str(e)}"

    def get_search_engines_resource(self) -> str:
        """
        MCP Resource: engines://list

        Current status of all configured search engines (modules + local database).
        Returns a markdown table of search engines status and capabilities.

        Returns:
            Markdown-formatted string containing search engines status
        """
        try:
            logger.info("Accessing search engines resource")

            # Get available engines
            engines = self.search_engine_manager.get_available_engines()

            if not engines:
                return "# Search Engines Status\n\nNo search engines available."

            # Format as markdown table
            markdown = "# Search Engines Status\n\n"
            markdown += "| Engine | Status | Description |\n"
            markdown += "|--------|--------|-------------|\n"

            for engine in engines:
                status = "✅ Available" if engine.is_available() else "❌ Unavailable"
                description = getattr(engine, 'description', f"{engine.name} search engine")
                markdown += f"| {engine.name} | {status} | {description} |\n"

            markdown += "\n"
            markdown += f"**Total Engines:** {len(engines)}\n\n"

            # Add engine capabilities summary
            markdown += "## Engine Capabilities\n\n"
            for engine in engines:
                markdown += f"### {engine.name}\n\n"
                if hasattr(engine, 'capabilities'):
                    capabilities = engine.capabilities
                    if isinstance(capabilities, list):
                        for cap in capabilities:
                            markdown += f"- {cap}\n"
                    else:
                        markdown += f"- {capabilities}\n"
                else:
                    markdown += "- Basic search functionality\n"
                markdown += "\n"

            return markdown

        except Exception as e:
            logger.error(f"Error accessing search engines resource: {e}")
            return "# Error Accessing Search Engines\n\nFailed to retrieve search engines status: {str(e)}"

    async def run(self, transport: str = "stdio", host: str = "0.0.0.0", port: int = 8000):
        """
        Run the MCP server with the specified transport.

        Args:
            transport: Transport method ('stdio', 'sse', 'http')
            host: Host for HTTP/SSE transports
            port: Port for HTTP/SSE transports
        """
        logger.info(f"Starting Research MCP Server with transport: {transport}")

        try:
            if transport == "stdio":
                await self.mcp_server.run_stdio_async()
            elif transport == "sse":
                await self.mcp_server.run_sse_async(host=host, port=port)
            elif transport == "http":
                await self.mcp_server.run_http_async(host=host, port=port)
            else:
                raise ValueError(f"Unsupported transport: {transport}")

        except Exception as e:
            logger.error(f"Server error: {e}")
            raise


async def main():
    """Main entry point for the Research MCP Server."""
    parser = argparse.ArgumentParser(description='Research Scholar MCP Server')
    parser.add_argument(
        '--directory',
        default='./knowledge-base',
        help='Directory for database and data files (default: ./knowledge-base)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--transport',
        choices=['stdio', 'sse', 'http'],
        default='stdio',
        help='Transport method (default: stdio)'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host for HTTP/SSE transports (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for HTTP/SSE transports (default: 8000)'
    )

    args = parser.parse_args()

    # Create and initialize the server
    server = ResearchMCPServer(
        data_directory=args.directory,
        verbose=args.verbose
    )

    # Initialize all components
    await server.initialize()

    # Run the server
    await server.run(
        transport=args.transport,
        host=args.host,
        port=args.port
    )


if __name__ == "__main__":
    # Run the server
    asyncio.run(main())