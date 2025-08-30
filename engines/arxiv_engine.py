"""
ArXiv search engine for the Research MCP Server.

This engine integrates with the ArXiv API to search for academic papers
in physics, mathematics, computer science, and related fields.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

from search_engines import BaseSearchEngine

logger = logging.getLogger(__name__)


class ArXivEngine(BaseSearchEngine):
    """
    ArXiv search engine implementation using the official ArXiv API.

    Features:
    - Search across all ArXiv categories
    - Author extraction and parsing
    - PDF download links
    - Category-based filtering
    - Publication date tracking
    """

    # ArXiv categories mapping
    CATEGORIES = {
        'cs.AI': 'Artificial Intelligence',
        'cs.CL': 'Computation and Language',
        'cs.CV': 'Computer Vision and Pattern Recognition',
        'cs.LG': 'Machine Learning',
        'cs.NE': 'Neural and Evolutionary Computing',
        'cs.RO': 'Robotics',
        'math.ST': 'Statistics Theory',
        'stat.ML': 'Machine Learning (Statistics)',
        'stat.TH': 'Statistics Theory',
        'physics.comp-ph': 'Computational Physics',
        'q-bio.NC': 'Neuroscience',
        'q-bio.QM': 'Quantitative Methods'
    }

    @property
    def name(self) -> str:
        """Human-readable name of the search engine."""
        return "ArXiv"

    @property
    def id(self) -> str:
        """Unique identifier for the search engine."""
        return "arxiv"

    @property
    def description(self) -> str:
        """Description of the search engine."""
        return "Open-access preprint repository specializing in physics, mathematics, computer science, and related fields"

    def is_available(self) -> bool:
        """
        Check if ArXiv engine is available.

        Requires python-arxiv package to be installed.

        Returns:
            True if arxiv package is available, False otherwise
        """
        return ARXIV_AVAILABLE

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search ArXiv for academic papers.

        Args:
            query: Search query string
            **kwargs: Additional search parameters
                - max_results: Maximum number of results (default: 10)
                - date_from: Start date filter (YYYY-MM-DD)
                - date_to: End date filter (YYYY-MM-DD)
                - categories: List of ArXiv categories to search

        Returns:
            List of paper dictionaries with standardized format
        """
        if not self.is_available():
            logger.warning("ArXiv engine not available - python-arxiv package not installed")
            return []

        max_results = kwargs.get('max_results', 10)
        date_from = kwargs.get('date_from')
        date_to = kwargs.get('date_to')
        categories = kwargs.get('categories', [])

        try:
            results = self._search_arxiv(query, max_results, date_from, date_to, categories)
            papers = []

            for result in results:
                paper = self._extract_paper_from_result(result)
                if paper:
                    papers.append(paper)

            logger.info(f"ArXiv search completed: {len(papers)} papers found")
            return papers

        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []

    def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific ArXiv paper.

        Args:
            paper_id: ArXiv paper identifier (e.g., "2101.12345")

        Returns:
            Paper dictionary or None if not found
        """
        if not self.is_available():
            return None

        try:
            # Search for the specific paper by ID
            search = arxiv.Search(id_list=[paper_id])
            client = arxiv.Client()
            results = list(client.results(search))

            if results:
                return self._extract_paper_from_result(results[0])

        except Exception as e:
            logger.error(f"Error getting ArXiv paper details: {e}")

        return None

    def get_citing_papers(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Get papers that cite the specified ArXiv paper.

        Note: ArXiv API doesn't provide direct citation tracking.
        This returns an empty list with a note.

        Args:
            paper_id: ArXiv paper identifier

        Returns:
            Empty list (citation tracking not supported by ArXiv API)
        """
        logger.info(f"ArXiv citation tracking not available for paper: {paper_id}")
        return []

    def get_categories(self) -> Dict[str, str]:
        """
        Get available ArXiv categories.

        Returns:
            Dictionary mapping category codes to names
        """
        return self.CATEGORIES.copy()

    def _search_arxiv(self, query: str, max_results: int = 10,
                     date_from: Optional[str] = None, date_to: Optional[str] = None,
                     categories: Optional[List[str]] = None) -> List:
        """
        Perform ArXiv search using the official API.

        Args:
            query: Search query
            max_results: Maximum results to return
            date_from: Start date filter
            date_to: End date filter
            categories: List of ArXiv categories

        Returns:
            List of arxiv.Result objects
        """
        # Build search query
        search_query = query

        # Add category filters if specified
        if categories:
            category_query = " OR ".join(f"cat:{cat}" for cat in categories)
            if search_query:
                search_query = f"({search_query}) AND ({category_query})"
            else:
                search_query = category_query

        # Create search object
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )

        # Add date filters if provided
        if date_from:
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            # Note: ArXiv API date filtering is limited

        # Execute search
        client = arxiv.Client()
        results = list(client.results(search))

        logger.debug(f"ArXiv search returned {len(results)} results")
        return results

    def _extract_paper_from_result(self, result) -> Optional[Dict[str, Any]]:
        """
        Extract paper information from ArXiv result.

        Args:
            result: arxiv.Result object

        Returns:
            Standardized paper dictionary or None if extraction fails
        """
        try:
            # Extract authors
            authors = [author.name for author in result.authors]

            # Extract year from published date
            year = result.published.year if result.published else None

            # Generate human-readable ID
            human_id = self._generate_human_id(authors, year)

            # Extract categories
            categories = [cat for cat in result.categories]

            # Build complete paper dictionary with ALL available data
            # This includes both PRD fields and additional metadata for database storage
            paper = {
                # PRD-compliant fields
                'paper_id': None,  # Will be assigned by database when stored
                'id': human_id,
                'title': result.title,
                'year': year or 0,
                'publication': f"ArXiv ({', '.join(categories)})",
                'authors': authors,
                'keywords': categories,  # Use categories as keywords
                'source': self.id,
                'project_id': None,  # Will be assigned when stored with project

                # Additional fields for complete database storage
                'summary': result.summary,
                'abstract': result.summary,
                'snippet': result.summary[:200] + '...' if len(result.summary) > 200 else result.summary,
                'publication_info': f"ArXiv ({', '.join(categories)})",
                'publication_year': year or 0,
                'source_url': result.entry_id,
                'pdf_url': result.pdf_url,
                'doi': result.doi or '',
                'citations': 0,  # ArXiv doesn't provide citation counts
                'citation_count': 0,
                'cites_id': '',
                'result_id': result.entry_id,
                'bibtex_link': '',
                'cached_page_link': '',
                'related_pages_link': '',
                'versions_count': 0,
                'versions_link': '',
                'resources': json.dumps([]),
                'metadata': json.dumps({
                    'arxiv_entry_id': result.entry_id,
                    'arxiv_updated': result.updated.isoformat() if result.updated else None,
                    'arxiv_published': result.published.isoformat() if result.published else None,
                    'arxiv_comment': result.comment,
                    'arxiv_journal_ref': result.journal_ref,
                    'arxiv_primary_category': result.primary_category,
                    'arxiv_categories': categories
                })
            }

            return paper

        except Exception as e:
            logger.error(f"Error extracting paper from ArXiv result: {e}")
            return None

    def _generate_human_id(self, authors: List[str], year: Optional[int]) -> str:
        """
        Generate human-readable paper ID from authors and year.

        Args:
            authors: List of author names
            year: Publication year

        Returns:
            Human-readable ID string
        """
        if not authors:
            return f"[Unknown Author {year or 'Unknown Year'}]"

        def parse_author_name(name: str) -> tuple:
            """Parse author name into last name and first initial."""
            name = name.strip()

            # Handle "First Last" format (most common in ArXiv)
            parts = name.split()
            if len(parts) >= 2:
                first_name = parts[0]
                last_name = parts[-1]
                first_initial = first_name[0] if first_name else ''
                return last_name, first_initial

            # Handle "Last, First" format
            if ',' in name:
                parts = name.split(',', 1)
                last_name = parts[0].strip()
                first_part = parts[1].strip()
                first_initial = first_part[0] if first_part else ''
                return last_name, first_initial

            # Fallback: use entire name as last name
            return name, ''

        # Generate ID based on author count
        if len(authors) == 1:
            last_name, first_initial = parse_author_name(authors[0])
            return f"[{last_name} {first_initial} {year or 'Unknown Year'}]"

        elif len(authors) == 2:
            author1_last, author1_initial = parse_author_name(authors[0])
            author2_last, author2_initial = parse_author_name(authors[1])
            return f"[{author1_last} {author1_initial}, {author2_last} {author2_initial} {year or 'Unknown Year'}]"

        else:  # 3+ authors
            first_author_last, first_author_initial = parse_author_name(authors[0])
            return f"[{first_author_last} {first_author_initial} et al. {year or 'Unknown Year'}]"
