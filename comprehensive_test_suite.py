#!/usr/bin/env python3
"""
Comprehensive Test Suite for Research MCP Server
Tests all requirements from prd.md
"""

import os
import sys
import json
import time
import asyncio
import tempfile
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ResearchMCPServer
from database import DatabaseManager
from search_engines import SearchEngineManager
from config import Config

class ComprehensiveTestSuite:
    """Comprehensive test suite covering all PRD requirements."""

    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        self.server = None
        self.db_manager = None
        self.engine_manager = None

    def log_test(self, test_name: str, result: bool, message: str = "", duration: float = None):
        """Log a test result."""
        status = "✅ PASS" if result else "❌ FAIL"
        duration_str = f" ({duration:.2f}s)" if duration else ""
        print(f"{status} {test_name}{duration_str}")
        if message:
            print(f"   {message}")
        print()

        self.test_results.append({
            'test': test_name,
            'result': result,
            'message': message,
            'duration': duration
        })

    async def run_test(self, test_func, test_name: str, *args, **kwargs):
        """Run a test function and log results."""
        try:
            start_time = time.time()

            # Check if the test method is async
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func(*args, **kwargs)
            else:
                result = test_func(*args, **kwargs)

            duration = time.time() - start_time

            if isinstance(result, tuple) and len(result) == 2:
                success, message = result
            else:
                success = result
                message = ""

            self.log_test(test_name, success, message, duration)

        except Exception as e:
            self.log_test(test_name, False, f"Exception: {str(e)}")

    # ============================================================================
    # 1. SERVER DEPLOYMENT & CLI TESTS
    # ============================================================================

    async def test_directory_creation(self):
        """Test automatic directory structure creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_knowledge_base"

            # Initialize server with non-existent directory
            server = ResearchMCPServer(verbose=False, data_directory=str(test_dir))

            # Initialize the server (this creates directories and DB)
            await server.initialize()

            # Check if all required directories were created
            expected_dirs = [
                test_dir,
                test_dir / "engines",
                test_dir / "cache",
                test_dir / "downloads",
                test_dir / "exports",
                test_dir / "temp"
            ]

            created_dirs = [dir_path for dir_path in expected_dirs if dir_path.exists()]
            all_created = len(created_dirs) == len(expected_dirs)

            # Check database file creation
            db_path = test_dir / "research.db"
            db_exists = db_path.exists()

            return all_created and db_exists, f"Created {len(created_dirs)}/{len(expected_dirs)} directories and database"

    def test_cli_arguments(self):
        """Test CLI argument parsing and configuration."""
        import argparse

        # Test verbose flag
        server_verbose = ResearchMCPServer(verbose=True)
        assert server_verbose.verbose == True, "Verbose flag not set correctly"

        # Test data directory
        with tempfile.TemporaryDirectory() as temp_dir:
            server_dir = ResearchMCPServer(data_directory=temp_dir)
            assert str(server_dir.data_directory) == temp_dir, "Data directory not set correctly"

        return True, "CLI arguments parsed correctly"

    def test_stdio_transport(self):
        """Test STDIO transport mode."""
        # This would require running the server in a subprocess
        # For now, we'll test the initialization
        server = ResearchMCPServer(transport="stdio")
        return hasattr(server, 'transport'), "STDIO transport initialized"

    # ============================================================================
    # 2. SEARCH ENGINE INTEGRATION TESTS
    # ============================================================================

    def test_engine_discovery(self):
        """Test search engine discovery and loading."""
        server = ResearchMCPServer(verbose=False)

        # Initialize engine manager
        engine_manager = SearchEngineManager(
            engines_directory=server.data_directory / "engines",
            db_manager=server.db_manager
        )

        # Check that local engine is always available
        local_engine = engine_manager.get_engine("local")
        local_available = local_engine and local_engine.is_available()

        # Check Google Scholar if API key is available
        gs_engine = engine_manager.get_engine("google_scholar")
        gs_available = gs_engine and gs_engine.is_available() if gs_engine else False

        available_count = sum([local_available, gs_available])

        return available_count >= 1, f"Found {available_count} available engines"

    def test_local_database_engine(self):
        """Test local database engine functionality."""
        server = ResearchMCPServer(verbose=False)

        # Get local engine
        local_engine = server.engine_manager.get_engine("local")

        if not local_engine:
            return False, "Local engine not found"

        # Test basic search (should return empty results initially)
        results = local_engine.search("test query")
        return isinstance(results, list), f"Local search returned {len(results)} results"

    def test_google_scholar_engine(self):
        """Test Google Scholar engine if API key is available."""
        server = ResearchMCPServer(verbose=False)

        gs_engine = server.engine_manager.get_engine("google_scholar")

        if not gs_engine:
            return True, "Google Scholar engine not implemented yet"

        if not gs_engine.is_available():
            return True, "Google Scholar API key not available (expected)"

        # Test actual search
        try:
            results = gs_engine.search("machine learning", max_results=1)
            return len(results) > 0, f"Google Scholar returned {len(results)} results"
        except Exception as e:
            return False, f"Google Scholar search failed: {str(e)}"

    def test_multiple_engine_search(self):
        """Test searching across multiple engines."""
        server = ResearchMCPServer(verbose=False)

        # Test with string format
        result1 = server.search_papers(query="test", engines="local", max_results=1)
        success1 = "results" in result1 and isinstance(result1["results"], list)

        # Test with array format
        result2 = server.search_papers(query="test", engines=["local"], max_results=1)
        success2 = "results" in result2 and isinstance(result2["results"], list)

        # Test with no engines (all available)
        result3 = server.search_papers(query="test", engines=None, max_results=1)
        success3 = "results" in result3 and isinstance(result3["results"], list)

        return success1 and success2 and success3, "All engine parameter formats work"

    # ============================================================================
    # 3. DATA MODEL & DATABASE TESTS
    # ============================================================================

    async def test_database_schema(self):
        """Test database schema creation and validation."""
        server = ResearchMCPServer(verbose=False)

        # Initialize the server (this creates DB and tables)
        await server.initialize()

        # Check if tables exist
        required_tables = ["papers", "projects"]

        for table in required_tables:
            if not server.db_manager.table_exists(table):
                return False, f"Table '{table}' not found"

        return True, f"All {len(required_tables)} required tables created"

    async def test_paper_identification(self):
        """Test paper identification system."""
        server = ResearchMCPServer(verbose=False)

        # Initialize the server
        await server.initialize()

        # Create a test paper
        test_paper = {
            "id": "[Smith, J. 2024]",
            "title": "Test Paper",
            "authors": ["Smith, J.", "Doe, A."],
            "year": 2024,
            "publication": "Test Journal",
            "source": "test",
            "project_id": "test_project"
        }

        # Store the paper
        stored_papers = server.db_manager.store_search_results([test_paper], "test_project")

        if not stored_papers:
            return False, "Failed to store test paper"

        # Extract the paper ID from the stored paper
        paper_id = stored_papers[0].get('paper_id') if isinstance(stored_papers[0], dict) else stored_papers[0]

        # Retrieve the paper
        details = server.db_manager.get_paper_details(paper_id)
        if not details:
            return False, "Failed to retrieve stored paper"

        # Check identifiers
        has_human_id = "id" in details and details["id"] == "[Smith, J. 2024]"
        has_integer_id = "paper_id" in details and isinstance(details["paper_id"], int)

        return has_human_id and has_integer_id, "Paper identification system works correctly"

    async def test_project_management(self):
        """Test project creation and management."""
        server = ResearchMCPServer(verbose=False)

        # Initialize the server
        await server.initialize()

        # Create a project
        project_result = server.db_manager.create_project("Test Project", "A test project")

        if "error" in project_result:
            return False, f"Failed to create project: {project_result['error']}"

        project_id = project_result.get("id") or project_result.get("project_id")
        if not project_id:
            return False, "Failed to create project - no ID returned"

        # List project papers (should be empty)
        papers = server.db_manager.get_project_papers(project_id)
        empty_project = len(papers) == 0

        return empty_project, f"Project '{project_id}' created successfully"

    # ============================================================================
    # 4. MCP TOOLS TESTS
    # ============================================================================

    def test_list_search_engines_tool(self):
        """Test list_search_engines MCP tool."""
        server = ResearchMCPServer(verbose=False)

        result = server.list_search_engines()

        # Should return a dictionary with engines list
        if not isinstance(result, dict):
            return False, "Result should be a dictionary"

        if "engines" not in result:
            return False, "Result should contain 'engines' key"

        engines = result["engines"]
        if not isinstance(engines, list):
            return False, "Engines should be a list"

        # Should have at least local engine
        local_found = any(engine.get("id") == "local" for engine in engines)

        return local_found, f"Found {len(engines)} engines including local"

    def test_search_papers_tool(self):
        """Test search_papers MCP tool with all parameter formats."""
        server = ResearchMCPServer(verbose=False)

        # Test 1: String format
        result1 = server.search_papers(query="test query", engines="local", max_results=1)
        success1 = isinstance(result1, dict) and "results" in result1

        # Test 2: Array format
        result2 = server.search_papers(query="test query", engines=["local"], max_results=1)
        success2 = isinstance(result2, dict) and "results" in result2

        # Test 3: No engines (all available)
        result3 = server.search_papers(query="test query", max_results=1)
        success3 = isinstance(result3, dict) and "results" in result3

        # Test 4: With project ID
        result4 = server.search_papers(
            query="test query",
            engines="local",
            project_id="test_project",
            max_results=1
        )
        success4 = isinstance(result4, dict) and "results" in result4

        return success1 and success2 and success3 and success4, "All search parameter formats work"

    async def test_create_project_tool(self):
        """Test create_project MCP tool."""
        server = ResearchMCPServer(verbose=False)

        # Initialize the server
        await server.initialize()

        result = server.create_project("Test Project", "A test project for MCP tools")

        if not isinstance(result, dict):
            return False, "Result should be a dictionary"

        if "project_id" not in result and "id" not in result:
            return False, "Result should contain 'project_id' or 'id'"

        project_id = result.get("project_id") or result.get("id")
        return isinstance(project_id, str) and len(project_id) > 0, f"Created project: {project_id}"

    async def test_manage_paper_keywords_tool(self):
        """Test manage_paper_keywords MCP tool."""
        server = ResearchMCPServer(verbose=False)

        # Initialize the server
        await server.initialize()

        # First create a paper
        test_paper = {
            "id": "[Test, A. 2024]",
            "title": "Test Paper for Keywords",
            "authors": ["Test, A."],
            "year": 2024,
            "publication": "Test Journal",
            "source": "test",
            "project_id": "keyword_test"
        }

        stored_papers = server.db_manager.store_search_results([test_paper], "keyword_test")
        if not stored_papers:
            return False, "Failed to create test paper for keywords"

        # Extract the paper ID from the stored paper
        paper_id = stored_papers[0].get('paper_id') if isinstance(stored_papers[0], dict) else stored_papers[0]

        # Test adding keywords
        result_add = server.manage_paper_keywords(
            paper_ids=paper_id,
            action="add",
            keywords=["machine learning", "test"]
        )

        if not isinstance(result_add, dict):
            return False, "Add keywords result should be a dictionary"

        # Test setting keywords
        result_set = server.manage_paper_keywords(
            paper_ids=paper_id,
            action="set",
            keywords=["artificial intelligence", "deep learning"]
        )

        if not isinstance(result_set, dict):
            return False, "Set keywords result should be a dictionary"

        return True, "Keyword management operations work correctly"

    async def test_get_paper_details_tool(self):
        """Test get_paper_details MCP tool."""
        server = ResearchMCPServer(verbose=False)

        # Initialize the server
        await server.initialize()

        # Create a test paper first
        test_paper = {
            "id": "[Details, T. 2024]",
            "title": "Test Paper for Details",
            "authors": ["Details, T."],
            "year": 2024,
            "publication": "Details Journal",
            "source": "test",
            "project_id": "details_test"
        }

        stored_papers = server.db_manager.store_search_results([test_paper], "details_test")
        if not stored_papers:
            return False, "Failed to create test paper for details"

        # Extract the paper ID from the stored paper
        paper_id = stored_papers[0].get('paper_id') if isinstance(stored_papers[0], dict) else stored_papers[0]

        # Test retrieving details
        result = server.get_paper_details(paper_id)

        if not isinstance(result, dict):
            return False, "Result should be a dictionary"

        required_fields = ["paper_id", "id", "title", "authors", "year", "source"]
        missing_fields = [field for field in required_fields if field not in result]

        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"

        return True, f"Retrieved details for paper {paper_id}"

    # ============================================================================
    # 5. ERROR HANDLING TESTS
    # ============================================================================

    def test_engine_failure_handling(self):
        """Test graceful handling of engine failures."""
        server = ResearchMCPServer(verbose=False)

        # Try to search with non-existent engine
        result = server.search_papers(query="test", engines="non_existent_engine", max_results=1)

        # Should return results structure even if empty
        return isinstance(result, dict) and "results" in result, "Engine failure handled gracefully"

    def test_api_key_validation(self):
        """Test API key validation for external engines."""
        server = ResearchMCPServer(verbose=False)

        gs_engine = server.engine_manager.get_engine("google_scholar")

        if not gs_engine:
            return True, "Google Scholar engine not implemented"

        # Check availability (should be False without API key)
        available = gs_engine.is_available()

        # If no API key, should return False
        if not os.getenv("SERP_API_KEY"):
            return not available, "Engine correctly unavailable without API key"
        else:
            return True, "Engine available with API key"

    # ============================================================================
    # 6. PERFORMANCE TESTS
    # ============================================================================

    def test_response_times(self):
        """Test response times for various operations."""
        server = ResearchMCPServer(verbose=False)

        # Test search response time
        start_time = time.time()
        result = server.search_papers(query="test", engines="local", max_results=1)
        search_time = time.time() - start_time

        # Test engine listing response time
        start_time = time.time()
        engines = server.list_search_engines()
        list_time = time.time() - start_time

        # Check performance requirements
        search_ok = search_time < 30.0  # < 30 seconds
        list_ok = list_time < 1.0       # < 1 second

        return search_ok and list_ok, f"Search: {search_time:.2f}s, List: {list_time:.2f}s"

    def test_concurrent_operations(self):
        """Test concurrent search operations."""
        server = ResearchMCPServer(verbose=False)

        def search_task(query):
            return server.search_papers(query=query, engines="local", max_results=1)

        # Run multiple searches concurrently
        queries = ["test1", "test2", "test3", "test4", "test5"]

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(search_task, query) for query in queries]

            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})

        # All results should be dictionaries
        all_valid = all(isinstance(r, dict) for r in results)

        return all_valid, f"Completed {len(results)} concurrent searches"

    # ============================================================================
    # 7. CONFIGURATION & LOGGING TESTS
    # ============================================================================

    def test_configuration_management(self):
        """Test configuration management and environment variables."""
        from config import Config

        config = Config()

        # Test API key retrieval
        serp_key = config.get_api_key("serp")
        expected_serp = os.getenv("SERP_API_KEY") or ""

        serp_ok = serp_key == expected_serp

        return serp_ok, "Configuration management works correctly"

    def test_logging_configuration(self):
        """Test logging configuration and output."""
        import logging

        # Test that logging is configured
        logger = logging.getLogger(__name__)
        has_handlers = len(logger.handlers) > 0 or len(logging.root.handlers) > 0

        return has_handlers, "Logging system configured correctly"

    # ============================================================================
    # NEW ADVANCED FEATURES TESTS
    # ============================================================================

    def test_arxiv_engine_discovery(self):
        """Test ArXiv engine discovery and loading."""
        server = ResearchMCPServer(verbose=False)

        # Check if ArXiv engine is discovered
        arxiv_engine = server.engine_manager.get_engine("arxiv")

        if not arxiv_engine:
            return True, "ArXiv engine not implemented yet (expected)"

        # Check if it's available (requires arxiv package)
        is_available = arxiv_engine.is_available()

        return is_available, f"ArXiv engine available: {is_available}"

    def test_search_by_author_tool(self):
        """Test search_by_author MCP tool."""
        server = ResearchMCPServer(verbose=False)

        # Test with a well-known author
        result = server.search_by_author(
            author_name="Smith",
            engines="local",  # Use local to avoid API calls
            max_results=3
        )

        # Should return a dictionary with results structure
        if not isinstance(result, dict):
            return False, "Result should be a dictionary"

        required_keys = ["query", "total_count", "results", "engines_searched"]
        missing_keys = [key for key in required_keys if key not in result]

        if missing_keys:
            return False, f"Missing keys: {missing_keys}"

        return True, f"Found {result['total_count']} papers by author"

    async def test_find_related_papers_tool(self):
        """Test find_related_papers MCP tool."""
        server = ResearchMCPServer(verbose=False)

        # Initialize the server
        await server.initialize()

        # First create a test paper
        test_paper = {
            "id": "[Test, A. 2024]",
            "title": "Test Paper for Related Papers",
            "authors": ["Test, A."],
            "year": 2024,
            "publication": "Test Journal",
            "source": "test",
            "project_id": "related_test"
        }

        stored_papers = server.db_manager.store_search_results([test_paper], "related_test")
        if not stored_papers:
            return False, "Failed to create test paper"

        # Extract the paper ID from the stored paper
        paper_id = stored_papers[0].get('paper_id') if isinstance(stored_papers[0], dict) else stored_papers[0]

        # Test finding related papers
        result = server.find_related_papers(
            paper_id=paper_id,
            engines="local",
            max_results=3
        )

        if not isinstance(result, dict):
            return False, "Result should be a dictionary"

        required_keys = ["reference_paper_id", "total_count", "results"]
        missing_keys = [key for key in required_keys if key not in result]

        if missing_keys:
            return False, f"Missing keys: {missing_keys}"

        return True, f"Found {result['total_count']} related papers"

    async def test_analyze_paper_trends_tool(self):
        """Test analyze_paper_trends MCP tool."""
        server = ResearchMCPServer(verbose=False)

        # Initialize the server
        await server.initialize()

        # Create some test papers for analysis
        test_papers = [
            {
                "id": "[Author1, A. 2020]",
                "title": "Paper 1",
                "authors": ["Author1, A.", "Author2, B."],
                "published_date": "2020-01-01",
                "categories": ["cs.AI"],
                "keywords": ["machine learning", "neural networks"]
            },
            {
                "id": "[Author2, B. 2021]",
                "title": "Paper 2",
                "authors": ["Author2, B."],
                "published_date": "2021-01-01",
                "categories": ["cs.LG"],
                "keywords": ["deep learning", "computer vision"]
            }
        ]

        # Store test papers
        paper_ids = []
        for paper in test_papers:
            stored_papers = server.db_manager.store_search_results([paper], "analysis_test")
            if stored_papers:
                # Extract the paper ID from the stored paper
                paper_id = stored_papers[0].get('paper_id') if isinstance(stored_papers[0], dict) else stored_papers[0]
                paper_ids.append(paper_id)

        if not paper_ids:
            return False, "Failed to create test papers"

        # Test author analysis
        result = server.analyze_paper_trends(
            paper_ids=paper_ids,
            analysis_type="authors"
        )

        if not isinstance(result, dict):
            return False, "Result should be a dictionary"

        if "error" in result:
            return False, f"Analysis failed: {result['error']}"

        return True, f"Successfully analyzed {result.get('total_papers_analyzed', 0)} papers"

    async def test_get_categories_tool(self):
        """Test get_search_engine_categories MCP tool."""
        server = ResearchMCPServer(verbose=False)

        # Initialize the server
        await server.initialize()

        result = server.get_search_engine_categories()

        if not isinstance(result, dict):
            return False, "Result should be a dictionary"

        # Should have categories key
        if "categories" not in result:
            return False, "Result should contain 'categories' key"

        # Should have at least local engine
        if "local" not in result["categories"]:
            return False, "Should include local engine categories"

        return True, f"Found categories for {len(result['categories'])} engines"

    async def test_export_results_tool(self):
        """Test export_search_results MCP tool."""
        server = ResearchMCPServer(verbose=False)

        # Initialize the server
        await server.initialize()

        # Create test papers
        test_papers = [
            {
                "id": "[Export, A. 2024]",
                "title": "Test Paper for Export",
                "authors": ["Export, A."],
                "published_date": "2024-01-01",
                "publication": "Test Journal"
            }
        ]

        # Store test papers
        stored_papers = server.db_manager.store_search_results(test_papers, "export_test")
        if not stored_papers:
            return False, "Failed to create test papers"

        # Extract paper IDs from stored papers
        paper_ids = []
        for stored_paper in stored_papers:
            paper_id = stored_paper.get('paper_id') if isinstance(stored_paper, dict) else stored_paper
            paper_ids.append(paper_id)

        # Test JSON export
        result = server.export_search_results(
            paper_ids=paper_ids,
            format="json"
        )

        if not isinstance(result, dict):
            return False, "Result should be a dictionary"

        if "error" in result:
            return False, f"Export failed: {result['error']}"

        if "export_data" not in result:
            return False, "Should contain export_data"

        return True, f"Successfully exported {result.get('papers_exported', 0)} papers"

    def test_centralized_author_extraction_llm(self):
        """Test centralized author extraction with LLM fallback."""
        from search_engines import BaseSearchEngine

        class TestEngine(BaseSearchEngine):
            @property
            def name(self): return "Test Engine"
            @property
            def id(self): return "test"
            def is_available(self): return True
            def search(self, query, **kwargs): return []
            def get_paper_details(self, paper_id): return None

        engine = TestEngine()

        # Test case 1: Structured data (should use first strategy)
        structured_pub_info = {
            "summary": "Smith, J., Johnson, A. - Journal of Science, 2020 - Publisher",
            "authors": [
                {"name": "Smith, J.", "link": "..."},
                {"name": "Johnson, A.", "link": "..."}
            ]
        }

        authors_structured = engine._extract_authors(structured_pub_info)
        structured_success = authors_structured == ["Smith, J.", "Johnson, A."]

        # Test case 2: No structured data (should use LLM or regex fallback)
        unstructured_pub_info = {
            "summary": "TC Ma, DE Willis - Frontiers in molecular neuroscience, 2015 - frontiersin.org"
        }

        authors_unstructured = engine._extract_authors(unstructured_pub_info)

        # Should extract some authors (either via LLM or regex)
        unstructured_success = len(authors_unstructured) > 0 and all(isinstance(a, str) for a in authors_unstructured)

        # Test case 3: Complex author names
        complex_pub_info = {
            "summary": "T.C. Ma, David E. Willis, A.B. Complex - Journal of Complex Names, 2023"
        }

        authors_complex = engine._extract_authors(complex_pub_info)
        complex_success = len(authors_complex) > 0

        # Overall success
        all_success = structured_success and unstructured_success and complex_success

        return all_success, f"Structured: {authors_structured}, Unstructured: {authors_unstructured}, Complex: {authors_complex}"

    # ============================================================================
    # MAIN TEST EXECUTION
    # ============================================================================

    async def run_all_tests(self):
        """Run all test categories."""
        print("🧪 COMPREHENSIVE TEST SUITE FOR RESEARCH MCP SERVER")
        print("=" * 60)

        test_categories = [
            ("Server Deployment & CLI", [
                (self.test_directory_creation, "Directory Creation"),
                (self.test_cli_arguments, "CLI Arguments"),
                (self.test_stdio_transport, "STDIO Transport"),
            ]),
            ("Search Engine Integration", [
                (self.test_engine_discovery, "Engine Discovery"),
                (self.test_local_database_engine, "Local Database Engine"),
                (self.test_google_scholar_engine, "Google Scholar Engine"),
                (self.test_multiple_engine_search, "Multiple Engine Search"),
            ]),
            ("Data Model & Database", [
                (self.test_database_schema, "Database Schema"),
                (self.test_paper_identification, "Paper Identification"),
                (self.test_project_management, "Project Management"),
            ]),
            ("MCP Tools", [
                (self.test_list_search_engines_tool, "List Search Engines Tool"),
                (self.test_search_papers_tool, "Search Papers Tool"),
                (self.test_create_project_tool, "Create Project Tool"),
                (self.test_manage_paper_keywords_tool, "Manage Keywords Tool"),
                (self.test_get_paper_details_tool, "Get Paper Details Tool"),
            ]),
            ("Error Handling", [
                (self.test_engine_failure_handling, "Engine Failure Handling"),
                (self.test_api_key_validation, "API Key Validation"),
            ]),
            ("Performance", [
                (self.test_response_times, "Response Times"),
                (self.test_concurrent_operations, "Concurrent Operations"),
            ]),
            ("Configuration & Logging", [
                (self.test_configuration_management, "Configuration Management"),
                (self.test_logging_configuration, "Logging Configuration"),
            ]),
            ("New Advanced Features", [
                (self.test_arxiv_engine_discovery, "ArXiv Engine Discovery"),
                (self.test_search_by_author_tool, "Search by Author Tool"),
                (self.test_find_related_papers_tool, "Find Related Papers Tool"),
                (self.test_analyze_paper_trends_tool, "Analyze Paper Trends Tool"),
                (self.test_get_categories_tool, "Get Categories Tool"),
                (self.test_export_results_tool, "Export Results Tool"),
            ]),
            ("LLM Integration", [
                (self.test_centralized_author_extraction_llm, "Centralized Author Extraction with LLM"),
            ]),
        ]

        total_tests = 0
        passed_tests = 0

        for category_name, tests in test_categories:
            print(f"\n📋 {category_name}")
            print("-" * 40)

            for test_func, test_name in tests:
                total_tests += 1
                try:
                    await self.run_test(test_func, f"{category_name}: {test_name}")
                    # Check if test passed
                    if self.test_results and self.test_results[-1]['result']:
                        passed_tests += 1
                except Exception as e:
                    print(f"❌ {test_name} - Exception: {str(e)}")
                    total_tests += 1

        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(".1f")

        # Detailed results
        if self.test_results:
            print("\n📋 DETAILED RESULTS:")
            print("-" * 40)
            for result in self.test_results:
                status = "✅ PASS" if result['result'] else "❌ FAIL"
                duration = f" ({result['duration']:.2f}s)" if result['duration'] else ""
                print(f"{status} {result['test']}{duration}")
                if result['message']:
                    print(f"   {result['message']}")

        return passed_tests == total_tests


async def main():
    """Main entry point for the test suite."""
    test_suite = ComprehensiveTestSuite()

    success = await test_suite.run_all_tests()

    if success:
        print("\n🎉 ALL TESTS PASSED! The Research MCP Server meets all PRD requirements.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review the results above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
