"""
Search engine management for the Research MCP Server.

This module handles:
- Discovery and loading of search engine modules
- Search engine availability checking
- Coordinated search across multiple engines
- Search result aggregation and formatting
"""

import importlib
import inspect
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from config import Config

logger = logging.getLogger(__name__)

# Optional imports for LLM functionality
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. LLM-enhanced parsing will be disabled.")


class BaseSearchEngine:
    """
    Base class for all search engines.

    All search engine modules must inherit from this class
    and implement the required methods.
    """

    @property
    def name(self) -> str:
        """Human-readable name of the search engine."""
        raise NotImplementedError

    @property
    def id(self) -> str:
        """Unique identifier for the search engine."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """
        Check if this search engine is available.

        Returns:
            True if the engine can be used, False otherwise
        """
        raise NotImplementedError

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for papers using this engine.

        Args:
            query: Search query string
            **kwargs: Additional search parameters

        Returns:
            List of paper dictionaries with standardized format
        """
        raise NotImplementedError

    def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific paper.

        Args:
            paper_id: Engine-specific paper identifier

        Returns:
            Paper dictionary or None if not found
        """
        raise NotImplementedError

    def get_citing_papers(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Get papers that cite the specified paper.

        Args:
            paper_id: Engine-specific paper identifier

        Returns:
            List of citing paper dictionaries
        """
        # Default implementation returns empty list
        # Engines that support citation tracking should override this
        return []

    # ===== CENTRALIZED PARSING METHODS =====

    def _extract_authors(self, pub_info: Dict) -> List[str]:
        """
        Extract author names using multiple fallback strategies.

        Strategy hierarchy:
        1. Structured data (authors array)
        2. LLM-enhanced parsing (if available)
        3. Regex-based text parsing
        4. Empty list as final fallback

        Args:
            pub_info: Publication info dictionary from API

        Returns:
            List of author names
        """
        # Strategy 1: Try structured data first
        authors = self._extract_authors_structured(pub_info)
        if authors:
            return authors

        # Strategy 2: Try LLM-enhanced parsing
        authors = self._extract_authors_with_llm(pub_info)
        if authors:
            return authors

        # Strategy 3: Fallback to regex-based text parsing
        authors = self._extract_authors_from_text(pub_info)
        if authors:
            return authors

        # Strategy 4: Return empty list
        return []

    def _extract_authors_structured(self, pub_info: Dict) -> List[str]:
        """
        Extract authors from structured data (authors array).

        Args:
            pub_info: Publication info dictionary

        Returns:
            List of author names or empty list if not available
        """
        authors_struct = pub_info.get("authors", [])

        if not authors_struct:
            return []

        authors = []
        for author in authors_struct:
            name = author.get("name", "").strip()
            if name:
                authors.append(name)

        return authors

    def _extract_authors_with_llm(self, pub_info: Dict) -> List[str]:
        """
        Extract authors using LLM when structured data isn't available.

        Args:
            pub_info: Publication info dictionary

        Returns:
            List of author names or empty list if LLM unavailable or fails
        """
        if not OPENAI_AVAILABLE:
            return []

        summary = pub_info.get("summary", "").strip()
        if not summary:
            return []

        try:
            # Get OpenAI API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.debug("OPENAI_API_KEY not found in environment")
                return []

            client = OpenAI(api_key=api_key)

            prompt = f"""
            Extract author names from this academic paper citation. Return only the author names as a JSON array.
            Keep the names in the same format as they appear in the citation.

            Citation: "{summary}"

            Examples:
            Input: "Smith, J., Johnson, A. - Journal of Science, 2020"
            Output: ["Smith, J.", "Johnson, A."]

            Input: "T.C. Ma, David E. Willis - Frontiers in Neuroscience, 2015"
            Output: ["T.C. Ma", "David E. Willis"]

            Input: "Ma, T.C., Willis, D.E. - Frontiers in Neuroscience, 2015"
            Output: ["Ma, T.C.", "Willis, D.E."]

            Return format: ["Author 1", "Author 2", ...]
            If no authors found, return: []
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )

            result_text = response.choices[0].message.content.strip()
            logger.debug(f"LLM raw response: {result_text}")

            # Try to parse as JSON
            try:
                import json
                authors = json.loads(result_text)
                if isinstance(authors, list):
                    # Clean and validate authors
                    clean_authors = []
                    for author in authors:
                        if isinstance(author, str) and author.strip():
                            clean_authors.append(author.strip())
                    logger.debug(f"LLM parsed authors: {clean_authors}")
                    return clean_authors
                else:
                    logger.debug(f"LLM returned non-list: {authors}")
            except json.JSONDecodeError as e:
                logger.debug(f"LLM returned non-JSON response: {result_text} (Error: {e})")

                # Try to extract authors from non-JSON response as fallback
                if result_text.startswith('[') and result_text.endswith(']'):
                    # Might be malformed JSON, try to fix it
                    try:
                        # Remove extra quotes or fix common issues
                        fixed_text = result_text.replace('""', '"')
                        authors = json.loads(fixed_text)
                        if isinstance(authors, list):
                            clean_authors = [a.strip() for a in authors if isinstance(a, str) and a.strip()]
                            logger.debug(f"LLM parsed fixed authors: {clean_authors}")
                            return clean_authors
                    except:
                        pass

        except Exception as e:
            logger.debug(f"LLM author extraction failed: {e}")

        return []

    def _extract_authors_from_text(self, pub_info: Dict) -> List[str]:
        """
        Extract authors from publication summary using regex patterns.

        Args:
            pub_info: Publication info dictionary

        Returns:
            List of author names or empty list if not found
        """
        summary = pub_info.get("summary", "").strip()
        if not summary:
            return []

        try:
            # Pattern 1: Extract text before first " - " separator
            # Example: "T.C. Ma, David E. Willis - Frontiers in Neuroscience, 2015"
            if " - " in summary:
                author_part = summary.split(" - ")[0].strip()

                # Split by common separators and clean up
                authors = []
                for part in author_part.split(","):
                    part = part.strip()
                    if part and len(part) > 1:  # Avoid single characters
                        # Clean up common artifacts
                        part = re.sub(r'[…\.\s]+$', '', part)  # Remove trailing ellipsis/dots
                        # Skip very short parts that are likely initials or fragments
                        if len(part) >= 3 and not part.isdigit():
                            authors.append(part)

                if authors:
                    return authors

            # Pattern 2: Look for author-like patterns in the entire summary
            # More conservative approach - look for proper name patterns
            author_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
            matches = re.findall(author_pattern, summary)

            if matches:
                # Filter out likely non-author matches
                filtered_matches = []
                for match in matches:
                    match = match.strip()
                    # Skip if it looks like a journal name (all caps, common journal words)
                    if (not re.match(r'^[A-Z\s]+$', match) and
                        len(match.split()) >= 2 and
                        not any(word.lower() in ['journal', 'science', 'review', 'research', 'university', 'institute']
                               for word in match.split())):
                        filtered_matches.append(match)

                return filtered_matches[:5]  # Limit to first 5 matches

        except Exception as e:
            logger.debug(f"Text-based author extraction failed: {e}")

        return []

    def _extract_year(self, text: str) -> Optional[int]:
        """
        Extract publication year from text using regex.

        Args:
            text: Text to search for year

        Returns:
            Extracted year or None if not found
        """
        if not text:
            return None

        # Match 4-digit years between 1900-2030
        match = re.search(r'\b(19|20)\d{2}\b', text)
        if match:
            year = int(match.group(0))
            # Sanity check
            if 1900 <= year <= 2030:
                return year

        return None


class LocalSearchEngine(BaseSearchEngine):
    """
    Local database search engine.

    Searches within the local database of stored papers.
    Always available since it doesn't require external APIs.
    """

    def __init__(self, db_manager):
        """
        Initialize local search engine.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self._name = "Local Database"
        self._id = "local"

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> str:
        return self._id

    def is_available(self) -> bool:
        """Local database is always available."""
        return True

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search within local database.

        Args:
            query: Search query
            **kwargs: Additional parameters (project_id, limit, etc.)

        Returns:
            List of paper dictionaries from local database
        """
        try:
            # Extract search parameters
            project_id = kwargs.get('project_id')
            limit = kwargs.get('max_results', 10)

            # For now, return empty list - this will be implemented
            # when we have the full search functionality in database.py
            logger.debug(f"Local search for query: '{query}' (project: {project_id})")
            return []

        except Exception as e:
            logger.error(f"Error in local search: {e}")
            return []

    def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get paper details from local database.

        Args:
            paper_id: Paper identifier

        Returns:
            Paper dictionary or None
        """
        try:
            return self.db_manager.get_paper_details(paper_id)
        except Exception as e:
            logger.error(f"Error getting local paper details: {e}")
            return None

    def get_citing_papers(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Get citing papers from local database.

        Args:
            paper_id: Paper identifier

        Returns:
            List of citing papers
        """
        try:
            return self.db_manager.get_citing_papers(paper_id)
        except Exception as e:
            logger.error(f"Error getting local citing papers: {e}")
            return []

    def search_by_author(self, author_name: str, **kwargs) -> List[Dict[str, Any]]:
        """Search papers by author in local database."""
        try:
            logger.debug(f"Local author search for: {author_name}")

            # Use database search functionality if available
            if hasattr(self.db_manager, 'search_by_author'):
                return self.db_manager.search_by_author(author_name, **kwargs)

            # Fallback: search in stored papers
            # This would require implementing author search in the database
            logger.info(f"Author search not yet implemented in local database for: {author_name}")
            return [{"error": f"Author search not yet implemented in {self.name}"}]

        except Exception as e:
            logger.error(f"Error in local author search: {e}")
            return []

    def find_related_papers(self, reference_paper: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Find related papers in local database."""
        try:
            logger.debug(f"Finding related papers for: {reference_paper.get('title', 'Unknown')}")

            # This would use keyword similarity and citation networks
            logger.info(f"Related papers search not yet implemented in {self.name}")
            return [{"error": f"Related papers search not yet implemented in {self.name}"}]

        except Exception as e:
            logger.error(f"Error finding related papers: {e}")
            return []

    def get_search_engine_categories(self) -> Dict[str, Any]:
        """Get local database categories - not applicable."""
        return {
            "engine_id": self.id,
            "engine_name": self.name,
            "categories": {},
            "total_categories": 0,
            "popular_categories": [],
            "usage_note": f"{self.name} doesn't use predefined categories. Search across all stored papers."
        }

    def analyze_paper_trends(self, papers: List[Dict[str, Any]], analysis_type: str) -> Dict[str, Any]:
        """Analyze paper trends in local database."""
        if not papers:
            return {"error": "No papers to analyze"}

        try:
            if analysis_type == "authors":
                # Analyze authors from provided papers
                all_authors = []
                for paper in papers:
                    authors = paper.get("authors", [])
                    if isinstance(authors, str):
                        authors = [authors]
                    all_authors.extend(authors)

                from collections import Counter
                author_counts = Counter(all_authors)

                return {
                    "analysis_type": "authors",
                    "total_unique_authors": len(author_counts),
                    "most_prolific_authors": author_counts.most_common(10),
                    "collaboration_stats": {
                        "avg_authors_per_paper": sum(len(p.get("authors", [])) for p in papers) / len(papers),
                        "single_author_papers": sum(1 for p in papers if len(p.get("authors", [])) == 1),
                        "multi_author_papers": sum(1 for p in papers if len(p.get("authors", [])) > 1),
                    },
                    "note": f"Analysis from {self.name} data"
                }

            elif analysis_type == "timeline":
                # Analyze publication timeline
                date_counts = {}
                for paper in papers:
                    date = paper.get("published_date", "")
                    if date:
                        year = date.split("-")[0]
                        date_counts[year] = date_counts.get(year, 0) + 1

                return {
                    "analysis_type": "timeline",
                    "papers_by_year": date_counts,
                    "most_active_year": max(date_counts.items(), key=lambda x: x[1]) if date_counts else None,
                    "total_years_span": len(date_counts),
                    "note": f"Timeline analysis from {self.name} data"
                }

            elif analysis_type == "categories":
                # Analyze categories if available
                category_counts = {}
                for paper in papers:
                    categories = paper.get("categories", [])
                    for cat in categories:
                        category_counts[cat] = category_counts.get(cat, 0) + 1

                return {
                    "analysis_type": "categories",
                    "total_categories": len(category_counts),
                    "most_common_categories": sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10],
                    "category_distribution": category_counts,
                    "note": f"Category analysis from {self.name} data"
                }

            elif analysis_type == "keywords":
                # Analyze keywords
                all_keywords = []
                for paper in papers:
                    keywords = paper.get("keywords", [])
                    if isinstance(keywords, str):
                        keywords = [keywords]
                    all_keywords.extend(keywords)

                from collections import Counter
                keyword_counts = Counter(all_keywords)

                return {
                    "analysis_type": "keywords",
                    "total_unique_keywords": len(keyword_counts),
                    "most_common_keywords": keyword_counts.most_common(20),
                    "keyword_distribution": dict(keyword_counts),
                    "note": f"Keyword analysis from {self.name} data"
                }

            else:
                return {"error": f"Unsupported analysis type: {analysis_type}"}

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def export_search_results(self, papers: List[Dict[str, Any]], format_type: str, **kwargs) -> Dict[str, Any]:
        """Export search results from local database."""
        try:
            import json
            from datetime import datetime

            if not papers:
                return {"error": "No papers to export"}

            if format_type == "json":
                content = json.dumps(papers, indent=4)
            elif format_type == "bibtex":
                # Generate BibTeX from stored paper data
                bibtex_entries = []
                export_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                header = f"""% Local Database Export
% Exported: {export_time}
"""
                bibtex_entries.append(header)

                for i, paper in enumerate(papers):
                    authors = paper.get("authors", ["unknown"])
                    if isinstance(authors, str):
                        authors = [authors]

                    year = paper.get("published_date", "unknown").split("-")[0]
                    title = paper.get("title", "No Title")

                    first_author = "unknown"
                    if authors and authors[0] != "unknown":
                        name_parts = authors[0].split(" ")
                        if name_parts:
                            first_author = name_parts[-1]

                    first_author = re.sub(r'[^a-zA-Z0-9]', '', first_author).lower()
                    key = f"{first_author}{year}"

                    # Handle duplicates
                    bibtex_keys = set()
                    original_key = key
                    suffix = 1
                    while key in bibtex_keys:
                        key = f"{original_key}_{suffix}"
                        suffix += 1
                    bibtex_keys.add(key)

                    author_str = " and ".join(authors)
                    pdf_url = paper.get("pdf_url", "")
                    journal = paper.get("publication", f"Local Database Paper {key}")

                    entry = f"""@article{{{key},
    title={{{title}}},
    author={{{author_str}}},
    journal={{{journal}}},
    year={{{year}}},
    url={{{pdf_url}}}
}}"""
                    bibtex_entries.append(entry)

                content = "\n\n".join(bibtex_entries)

            elif format_type == "csv":
                try:
                    import pandas as pd
                    df = pd.DataFrame(papers)
                    content = df.to_string()
                except ImportError:
                    return {"error": "CSV export requires pandas library"}

            elif format_type == "markdown":
                md_entries = []
                for paper in papers:
                    title = paper.get("title", "N/A")
                    authors = ", ".join(paper.get("authors", ["N/A"])) if isinstance(paper.get("authors"), list) else paper.get("authors", "N/A")
                    date = paper.get("published_date", "N/A")
                    url = paper.get("pdf_url", "#")
                    summary = paper.get("summary", "N/A").replace("\n", " ")

                    md_entries.append(f"""### {title}\n**Authors:** {authors}\n**Published:** {date}\n**[PDF Link]({url})**\n> {summary}\n""")

                content = "\n---\n".join(md_entries)

            else:
                return {"error": f"Unsupported format: {format_type}"}

            return {
                "success": True,
                "format": format_type,
                "papers_exported": len(papers),
                "content_preview": content[:500] + ("..." if len(content) > 500 else ""),
                "export_data": content,
                "note": f"Export from {self.name}"
            }

        except Exception as e:
            return {"error": f"Export failed: {str(e)}"}


class SearchEngineManager:
    """
    Manager for all search engines.

    Handles discovery, loading, and coordination of search engines.
    """

    def __init__(self, engines_directory: Path, db_manager=None):
        """
        Initialize search engine manager.

        Args:
            engines_directory: Directory containing search engine modules
            db_manager: Database manager instance for local search
        """
        self.engines_directory = engines_directory
        self.db_manager = db_manager
        self.config = Config()

        # Dictionary to store loaded engines (only available ones)
        self._engines: Dict[str, BaseSearchEngine] = {}

        # List to store all discovered engines (including unavailable ones)
        self._all_discovered_engines: List[BaseSearchEngine] = []

        # Initialize built-in engines
        self._initialize_builtin_engines()

        logger.info(f"Search engine manager initialized with directory: {engines_directory}")

    def _initialize_builtin_engines(self):
        """Initialize built-in search engines."""
        # Add local database engine
        if self.db_manager:
            local_engine = LocalSearchEngine(self.db_manager)
            self._engines[local_engine.id] = local_engine
            self._all_discovered_engines.append(local_engine)
            logger.debug(f"Initialized built-in engine: {local_engine.name}")

    async def discover_engines(self):
        """
        Discover and load search engine modules from the engines directory.

        This method:
        1. Scans the engines directory for Python files
        2. Imports each module
        3. Instantiates search engine classes
        4. Validates engine availability
        5. Registers available engines
        """
        logger.info("Discovering search engines...")
        logger.debug(f"Engines directory: {self.engines_directory}")
        logger.debug(f"Current working directory: {os.getcwd()}")
        logger.debug(f"Python path: {sys.path[:3]}...")  # Show first 3 paths

        if not self.engines_directory.exists():
            logger.warning(f"Engines directory does not exist: {self.engines_directory}")
            return

        # Add engines directory to Python path
        if str(self.engines_directory) not in sys.path:
            sys.path.insert(0, str(self.engines_directory))
            logger.debug(f"Added {self.engines_directory} to Python path")

        # Scan for Python files
        engine_files = list(self.engines_directory.glob("*.py"))
        logger.debug(f"Found {len(engine_files)} Python files: {[f.name for f in engine_files]}")

        for engine_file in engine_files:
            if engine_file.name.startswith('_'):
                continue  # Skip private modules

            try:
                module_name = engine_file.stem
                logger.debug(f"Attempting to import module: {module_name}")

                # Import the module
                module = importlib.import_module(module_name)
                logger.debug(f"Successfully imported module: {module_name}")

                # Find search engine classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                        issubclass(obj, BaseSearchEngine) and
                        obj != BaseSearchEngine):
                        logger.debug(f"Found search engine class: {name}")

                        try:
                            # Instantiate the engine
                            logger.debug(f"Instantiating engine class: {name}")
                            if hasattr(obj, '__init__') and 'db_manager' in inspect.signature(obj.__init__).parameters:
                                engine = obj(self.db_manager)
                            else:
                                engine = obj()

                            logger.debug(f"Engine instantiated: {engine.name} (ID: {engine.id})")

                            # Check availability
                            logger.debug(f"Checking availability for engine: {engine.name}")
                            is_available = engine.is_available()
                            logger.debug(f"Engine {engine.name} availability: {is_available}")

                            # Additional debugging for Google Scholar
                            if engine.id == 'google_scholar':
                                logger.debug(f"Environment variables during discovery:")
                                logger.debug(f"  SERP_API_KEY set: {bool(os.getenv('SERP_API_KEY'))}")
                                logger.debug(f"  All env vars with 'SERP': {[k for k in os.environ.keys() if 'SERP' in k]}")
                                logger.debug(f"  All env vars with 'API': {[k for k in os.environ.keys() if 'API' in k]}")

                            # Store all discovered engines (including unavailable ones)
                            self._all_discovered_engines.append(engine)

                            if is_available:
                                self._engines[engine.id] = engine
                                logger.info(f"Loaded search engine: {engine.name} (ID: {engine.id})")
                            else:
                                logger.debug(f"Search engine {engine.name} is not available (skipping)")

                        except Exception as e:
                            logger.error(f"Error instantiating engine {name}: {e}")

            except Exception as e:
                logger.error(f"Error loading engine module {engine_file}: {e}")

        logger.info(f"Search engine discovery complete. Loaded {len(self._engines)} engines")

    def get_available_engines(self) -> List[BaseSearchEngine]:
        """
        Get list of all available search engines.

        Returns:
            List of available search engine instances
        """
        return list(self._engines.values())

    def get_all_discovered_engines(self) -> List[BaseSearchEngine]:
        """
        Get list of all discovered search engines (including unavailable ones).

        Returns:
            List of all discovered search engine instances
        """
        return self._all_discovered_engines.copy()

    def get_engine(self, engine_id: str) -> Optional[BaseSearchEngine]:
        """
        Get a specific search engine by ID.

        Args:
            engine_id: Engine identifier

        Returns:
            Search engine instance or None if not found
        """
        return self._engines.get(engine_id)

    def search_papers(
        self,
        query: str,
        engines: Optional[Union[str, List[str]]] = None,
        max_results: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for papers across specified engines.

        Args:
            query: Search query
            engines: Single engine name (string) or list of engine IDs to search (None for all available)
            max_results: Maximum results per engine
            **kwargs: Additional search parameters

        Returns:
            List of paper dictionaries from all engines
        """
        logger.info(f"Searching for papers with query: '{query}'")

        all_results = []

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

        # Determine which engines to use
        if normalized_engines is None:
            search_engines = self.get_available_engines()
        else:
            search_engines = []
            for engine_id in normalized_engines:
                engine = self.get_engine(engine_id)
                if engine and engine.is_available():
                    search_engines.append(engine)
                else:
                    logger.warning(f"Engine '{engine_id}' not found or not available")

        # Search each engine
        for engine in search_engines:
            try:
                logger.debug(f"Searching {engine.name} for: '{query}'")

                engine_results = engine.search(
                    query=query,
                    max_results=max_results,
                    **kwargs
                )

                # Add source information to results
                for result in engine_results:
                    result['source'] = engine.id
                    result['_engine_name'] = engine.name

                all_results.extend(engine_results)
                logger.debug(f"Found {len(engine_results)} results from {engine.name}")

            except Exception as e:
                logger.error(f"Error searching {engine.name}: {e}")
                # Continue with other engines

        logger.info(f"Total results from all engines: {len(all_results)}")
        return all_results

    def get_engine_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all engines.

        Returns:
            Dictionary mapping engine IDs to status information
        """
        status = {}

        for engine_id, engine in self._engines.items():
            status[engine_id] = {
                'name': engine.name,
                'id': engine.id,
                'available': engine.is_available(),
                'type': 'builtin' if engine_id == 'local' else 'external'
            }

        return status

    def reload_engines(self):
        """
        Reload all search engines.

        This can be useful for development or when engine configurations change.
        """
        logger.info("Reloading search engines...")

        # Clear existing engines
        self._engines.clear()

        # Re-initialize built-in engines
        self._initialize_builtin_engines()

        # Re-discover external engines
        import asyncio
        asyncio.create_task(self.discover_engines())

        logger.info("Search engines reloaded")
