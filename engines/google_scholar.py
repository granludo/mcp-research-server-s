"""
Google Scholar search engine for the Research MCP Server.

This engine integrates with SerpAPI to search Google Scholar for academic papers.
It provides comprehensive search capabilities including citation tracking,
author extraction, and metadata parsing.
"""

import json
import logging
import re
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

from search_engines import BaseSearchEngine

logger = logging.getLogger(__name__)


class GoogleScholarEngine(BaseSearchEngine):
    """
    Google Scholar search engine implementation using SerpAPI.

    Features:
    - Academic paper search via Google Scholar
    - Citation tracking and citing papers
    - Author extraction with LLM enhancement
    - Publication metadata parsing
    - PDF link discovery
    """

    @property
    def name(self) -> str:
        """Human-readable name of the search engine."""
        return "Google Scholar"

    @property
    def id(self) -> str:
        """Unique identifier for the search engine."""
        return "google_scholar"

    @property
    def description(self) -> str:
        """Description of the search engine."""
        return "Comprehensive academic search engine with citation metrics and broad coverage across all academic disciplines"

    def is_available(self) -> bool:
        """
        Check if Google Scholar engine is available.

        Requires SERP_API_KEY environment variable.

        Returns:
            True if SerpAPI key is available, False otherwise
        """
        import os
        api_key = os.getenv("SERP_API_KEY")
        return api_key is not None

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search Google Scholar for academic papers.

        Args:
            query: Search query string
            **kwargs: Additional search parameters
                - max_results: Maximum number of results (default: 10)
                - date_from: Start date filter (YYYY-MM-DD)
                - date_to: End date filter (YYYY-MM-DD)

        Returns:
            List of paper dictionaries with standardized format
        """
        if not self.is_available():
            logger.warning("Google Scholar engine not available - SERP_API_KEY not found")
            return []

        max_results = kwargs.get('max_results', 10)
        date_from = kwargs.get('date_from')
        date_to = kwargs.get('date_to')

        try:
            results = self._search_google_scholar(query, max_results, date_from, date_to)
            papers = []

            for result in results:
                paper = self._extract_paper_from_result(result)
                if paper:
                    papers.append(paper)

            logger.info(f"Google Scholar search completed: {len(papers)} papers found")
            return papers

        except Exception as e:
            logger.error(f"Google Scholar search failed: {e}")
            return []

    def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific paper.

        Note: SerpAPI doesn't provide direct paper detail lookup by ID,
        so this returns basic information if available from cache.

        Args:
            paper_id: Paper identifier (not directly usable with SerpAPI)

        Returns:
            None (not supported by this engine)
        """
        logger.debug(f"Paper detail lookup not supported for Google Scholar engine")
        return None

    def get_citing_papers(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Get papers that cite the specified paper.

        Args:
            paper_id: Paper identifier from Google Scholar

        Returns:
            List of citing paper dictionaries
        """
        if not self.is_available():
            logger.warning("Google Scholar engine not available for citation search")
            return []

        try:
            # Use SerpAPI's cite parameter to get citing papers
            citing_results = self._search_google_scholar_with_cites(paper_id)
            citing_papers = []

            for result in citing_results:
                paper = self._extract_paper_from_result(result)
                if paper:
                    # Mark as citing paper
                    paper['is_citing_paper'] = True
                    paper['cited_paper_id'] = paper_id
                    citing_papers.append(paper)

            logger.info(f"Found {len(citing_papers)} citing papers")
            return citing_papers

        except Exception as e:
            logger.error(f"Error getting citing papers: {e}")
            return []

    def _search_google_scholar(self, query: str, max_results: int = 10,
                              date_from: Optional[str] = None,
                              date_to: Optional[str] = None) -> List[Dict]:
        """
        Perform Google Scholar search using SerpAPI.

        Args:
            query: Search query
            max_results: Maximum results to return
            date_from: Start date filter
            date_to: End date filter

        Returns:
            Raw API response results
        """
        import os

        api_key = os.getenv("SERP_API_KEY")
        if not api_key:
            raise ValueError("SERP_API_KEY environment variable not set")

        # Build search parameters
        params = {
            'engine': 'google_scholar',
            'q': query,
            'api_key': api_key,
            'num': min(max_results, 20),  # SerpAPI max is 20
            'start': 0
        }

        # Add date filters if provided
        if date_from or date_to:
            if date_from:
                params['as_ylo'] = date_from.split('-')[0]  # Year only
            if date_to:
                params['as_yhi'] = date_to.split('-')[0]    # Year only

        # Make API request
        url = f"https://serpapi.com/search.json?{urlencode(params)}"

        logger.debug(f"Making SerpAPI request: {url.replace(api_key, '***')}")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Extract organic results
        organic_results = data.get('organic_results', [])
        logger.debug(f"Received {len(organic_results)} organic results from SerpAPI")

        return organic_results

    def _search_google_scholar_with_cites(self, paper_id: str) -> List[Dict]:
        """
        Search for papers citing a specific paper using SerpAPI's cite parameter.

        Args:
            paper_id: Google Scholar paper identifier

        Returns:
            List of citing paper results
        """
        import os

        api_key = os.getenv("SERP_API_KEY")
        if not api_key:
            raise ValueError("SERP_API_KEY environment variable not set")

        # Build citation search parameters
        params = {
            'engine': 'google_scholar',
            'q': '',  # Empty query
            'api_key': api_key,
            'num': 10,
            'cites': paper_id  # Cite this paper
        }

        url = f"https://serpapi.com/search.json?{urlencode(params)}"

        logger.debug(f"Making citation SerpAPI request for paper: {paper_id}")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        organic_results = data.get('organic_results', [])

        return organic_results

    def _extract_paper_from_result(self, result: Dict) -> Optional[Dict[str, Any]]:
        """
        Extract paper information from SerpAPI result.

        Args:
            result: Single result from SerpAPI response

        Returns:
            Standardized paper dictionary or None if extraction fails
        """
        try:
            # Extract basic information
            title = result.get('title', '').strip()
            if not title:
                return None

            # Extract publication info and authors
            publication_info = result.get('publication_info', {})
            summary = publication_info.get('summary', '')

            # Extract authors using centralized method
            authors = self._extract_authors(publication_info)

            # Extract year
            year = self._extract_year(summary)

            # Generate human-readable ID
            human_id = self._generate_human_id(authors, year)

            # Extract other metadata
            link = result.get('link', '')
            snippet = result.get('snippet', '')

            # Extract keywords from available text
            keywords = self._extract_keywords(title, snippet, summary)
            pdf_link = self._find_pdf_url(result)

            # Build complete paper dictionary with ALL available data
            # This includes both PRD fields and additional metadata for database storage
            paper = {
                # PRD-compliant fields
                'paper_id': None,  # Will be assigned by database when stored
                'id': human_id,
                'title': title,
                'year': year or 0,
                'publication': self._extract_publication(summary),
                'authors': authors,
                'keywords': keywords,
                'source': self.id,
                'project_id': None,  # Will be assigned when stored with project

                # Additional fields for complete database storage
                'summary': summary,
                'abstract': snippet,
                'snippet': snippet,
                'publication_info': self._extract_publication(summary),
                'publication_year': year or 0,
                'source_url': link,
                'pdf_url': pdf_link,
                'doi': '',
                'citations': self._extract_citation_count(result),
                'citation_count': self._extract_citation_count(result),
                'cites_id': result.get('cited_by_url', ''),
                'result_id': result.get('result_id', ''),
                'bibtex_link': '',
                'cached_page_link': result.get('cached_page_link', ''),
                'related_pages_link': result.get('related_pages_link', ''),
                'versions_count': result.get('versions', {}).get('count', 0),
                'versions_link': result.get('versions', {}).get('link', ''),
                'resources': json.dumps(result.get('resources', [])),
                'metadata': json.dumps({
                    'serpapi_result': result,
                    'extraction_timestamp': str(datetime.now())
                })
            }

            return paper

        except Exception as e:
            logger.error(f"Error extracting paper from result: {e}")
            return None

    def _extract_keywords(self, title: str, snippet: str, summary: str) -> List[str]:
        """
        Extract keywords from paper text fields.

        Args:
            title: Paper title
            snippet: Paper snippet/abstract
            summary: Publication summary

        Returns:
            List of extracted keywords
        """
        # Combine all text for keyword extraction
        text = f"{title} {snippet} {summary}".lower()

        # Academic/ML keywords to look for
        academic_keywords = [
            # Machine Learning & AI
            'machine learning', 'deep learning', 'neural network', 'artificial intelligence', 'ai',
            'computer vision', 'natural language processing', 'nlp', 'reinforcement learning',
            'supervised learning', 'unsupervised learning', 'semi-supervised learning',

            # Algorithms & Techniques
            'classification', 'regression', 'clustering', 'optimization', 'gradient descent',
            'backpropagation', 'convolutional neural network', 'cnn', 'recurrent neural network', 'rnn',
            'transformer', 'attention mechanism', 'embedding', 'feature extraction', 'dimensionality reduction',

            # Computer Science
            'algorithm', 'data mining', 'big data', 'data science', 'parallel computing',
            'distributed computing', 'cloud computing', 'edge computing',

            # Research Areas
            'information retrieval', 'search engine', 'recommendation system', 'chatbot',
            'virtual assistant', 'autonomous system', 'robotics', 'computer graphics',
            'human-computer interaction', 'software engineering',

            # Data & Statistics
            'statistics', 'probability', 'bayesian', 'stochastic process', 'time series',
            'anomaly detection', 'predictive modeling', 'data visualization',

            # Applications
            'medical imaging', 'drug discovery', 'financial modeling', 'cybersecurity',
            'climate modeling', 'autonomous vehicle', 'smart city', 'internet of things',

            # General Academic
            'research', 'methodology', 'evaluation', 'benchmark', 'performance',
            'scalability', 'efficiency', 'robustness', 'generalization'
        ]

        extracted_keywords = []

        for keyword in academic_keywords:
            if keyword in text:
                # Capitalize first letter of each word for display
                formatted_keyword = ' '.join(word.capitalize() for word in keyword.split())
                if formatted_keyword not in extracted_keywords:
                    extracted_keywords.append(formatted_keyword)

        # Limit to top 10 most relevant keywords
        return extracted_keywords[:10]

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

            # Handle "Last, First" format
            if ',' in name:
                parts = name.split(',', 1)
                last_name = parts[0].strip()
                first_part = parts[1].strip()
                first_initial = first_part[0] if first_part else ''
                return last_name, first_initial

            # Handle "First Last" format
            parts = name.split()
            if len(parts) >= 2:
                first_name = parts[0]
                last_name = parts[-1]
                first_initial = first_name[0] if first_name else ''
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

    def _extract_publication(self, summary: str) -> str:
        """
        Extract publication information from summary text.

        Args:
            summary: Publication summary text

        Returns:
            Publication string
        """
        if not summary:
            return "Unknown Publication"

        # Try to extract publication after " - " separator
        if " - " in summary:
            parts = summary.split(" - ", 1)
            if len(parts) > 1:
                publication_part = parts[1].strip()
                # Remove year if present at the end
                publication_part = re.sub(r',\s*\d{4}$', '', publication_part)
                return publication_part

        return "Unknown Publication"

    def _find_pdf_url(self, result: Dict) -> Optional[str]:
        """
        Find PDF download URL from result.

        Args:
            result: SerpAPI result dictionary

        Returns:
            PDF URL or None if not found
        """
        # Check for direct PDF link
        resources = result.get('resources', [])
        for resource in resources:
            if resource.get('file_format') == 'PDF':
                return resource.get('link')

        # Check for PDF in snippet or other fields
        snippet = result.get('snippet', '').lower()
        if 'pdf' in snippet:
            # Look for PDF links in the result
            for key, value in result.items():
                if isinstance(value, str) and value.lower().endswith('.pdf'):
                    return value

        return None

    def _extract_citation_count(self, result: Dict) -> Optional[int]:
        """
        Extract citation count from result.

        Args:
            result: SerpAPI result dictionary

        Returns:
            Citation count or None if not found
        """
        cited_by = result.get('cited_by', {})
        if isinstance(cited_by, dict):
            count = cited_by.get('value')
            if isinstance(count, str):
                # Extract number from strings like "Cited by 42"
                match = re.search(r'(\d+)', count)
                if match:
                    return int(match.group(1))
            elif isinstance(count, int):
                return count

        return None
