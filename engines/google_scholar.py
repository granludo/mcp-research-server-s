"""
Google Scholar search engine for the Research MCP Server.

Based on sample_scholar_query.py, this module provides Google Scholar
search capabilities using the SerpAPI service.

Features:
- Comprehensive paper metadata extraction
- Author profile information
- Citation counts and links
- BibTeX download links
- PDF and resource discovery
"""

import os
import re
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, parse_qs

import requests
from config import Config
from search_engines import BaseSearchEngine

logger = logging.getLogger(__name__)


class GoogleScholarEngine(BaseSearchEngine):
    """
    Google Scholar search engine using SerpAPI.

    This engine provides comprehensive search capabilities
    for Google Scholar with rich metadata extraction.
    """

    def __init__(self):
        """Initialize Google Scholar engine."""
        self.config = Config()
        self.api_key = self.config.get_api_key('google_scholar')
        self.endpoint = "https://serpapi.com/search.json"

        self._name = "Google Scholar"
        self._id = "google_scholar"

    @property
    def name(self) -> str:
        """Human-readable name of the search engine."""
        return self._name

    @property
    def id(self) -> str:
        """Unique identifier for the search engine."""
        return self._id

    def is_available(self) -> bool:
        """
        Check if Google Scholar engine is available.

        Returns:
            True if SERP_API_KEY is set, False otherwise
        """
        import os
        self.api_key = self.config.get_api_key('google_scholar')

        logger.debug(f"Google Scholar engine checking availability...")
        logger.debug(f"SERP_API_KEY environment variable set: {bool(os.getenv('SERP_API_KEY'))}")
        logger.debug(f"Config api_key value: {'***' + self.api_key[-4:] if self.api_key else 'None'}")
        logger.debug(f"Final availability result: {bool(self.api_key)}")

        return bool(self.api_key)

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search Google Scholar for papers.

        Args:
            query: Search query
            **kwargs: Additional search parameters (max_results, etc.)

        Returns:
            List of paper dictionaries with standardized format
        """
        try:
            max_results = kwargs.get('max_results', 5)

            # Prepare search parameters
            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": self.api_key,
                "num": min(max_results, 20)  # SerpAPI limit
            }

            logger.debug(f"Searching Google Scholar for: '{query}'")

            # Make API request
            response = requests.get(
                self.endpoint,
                params=params,
                timeout=self.config.get('request_timeout', 10.0)
            )
            response.raise_for_status()

            data = response.json()

            # Extract papers from results
            papers = self._extract_papers_from_results(data, query)

            logger.info(f"Found {len(papers)} papers from Google Scholar")
            return papers

        except requests.RequestException as e:
            logger.error(f"Google Scholar API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {e}")
            return []

    def _extract_papers_from_results(self, data: Dict, query: str) -> List[Dict[str, Any]]:
        """
        Extract paper information from SerpAPI response.

        Args:
            data: SerpAPI response data
            query: Original search query

        Returns:
            List of standardized paper dictionaries
        """
        papers = []

        organic_results = data.get("organic_results", [])

        for i, result in enumerate(organic_results):
            try:
                paper = self._extract_paper_from_result(result, i)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Error extracting paper {i}: {e}")
                continue

        return papers

    def _extract_paper_from_result(self, result: Dict, index: int) -> Optional[Dict[str, Any]]:
        """
        Extract paper information from a single SerpAPI result.

        Args:
            result: Single result from SerpAPI
            index: Result index for generating IDs

        Returns:
            Standardized paper dictionary or None if extraction fails
        """
        try:
            # Basic information
            title = result.get("title", "")
            link = result.get("link", "")
            snippet = result.get("snippet", "")

            # Publication information
            pub_info = result.get("publication_info", {})
            summary = pub_info.get("summary", "")
            authors = self._extract_authors(pub_info)  # Now uses centralized parsing

            # Extract year from publication info
            year = self._extract_year(summary)  # Now uses centralized parsing

            # Citation information
            inline_links = result.get("inline_links", {})
            cited_by = inline_links.get("cited_by", {})
            citation_count = cited_by.get("total", 0)
            cites_id = cited_by.get("cites_id")

            # Resource links
            resources = result.get("resources", [])
            pdf_url = self._find_pdf_url(resources)

            # Generate result ID
            result_id = result.get("result_id", f"scholar_{index}")

            # Generate human-readable ID
            logger.debug(f"Generating ID for authors: {authors}, year: {year}")
            if authors and len(authors) > 0:
                # Create a more comprehensive author ID
                def parse_author_name(author):
                    """Parse author name into first initial and last name."""
                    author = author.strip()
                    logger.debug(f"Parsing author name: '{author}'")
                    if ',' in author:
                        # Format: "Last, First" or "Last, F."
                        parts = author.split(',')
                        last_name = parts[0].strip()
                        first_part = parts[1].strip() if len(parts) > 1 else ''
                        first_initial = first_part[0] if first_part else ''
                        logger.debug(f"Parsed as Last,First: last='{last_name}', first='{first_initial}'")
                        return last_name, first_initial
                    else:
                        # Format: "First Last" or "F. Last"
                        parts = author.split()
                        if len(parts) >= 2:
                            first_part = parts[0]
                            last_name = parts[-1]
                            first_initial = first_part[0] if first_part else ''
                            logger.debug(f"Parsed as First Last: last='{last_name}', first='{first_initial}'")
                            return last_name, first_initial
                        else:
                            # Single word, assume it's last name
                            logger.debug(f"Single word author: '{author}'")
                            return author, ''

                if len(authors) == 1:
                    # Single author: [LastName FirstInitial Year]
                    last_name, first_initial = parse_author_name(authors[0])
                    if first_initial:
                        human_id = f"[{last_name} {first_initial} {year or 'Unknown'}]"
                    else:
                        human_id = f"[{last_name} {year or 'Unknown'}]"
                elif len(authors) == 2:
                    # Two authors: [LastName1 FI, LastName2 FI Year]
                    author1_last, author1_first = parse_author_name(authors[0])
                    author2_last, author2_first = parse_author_name(authors[1])

                    author1_part = f"{author1_last} {author1_first}".strip()
                    author2_part = f"{author2_last} {author2_first}".strip()

                    human_id = f"[{author1_part}, {author2_part} {year or 'Unknown'}]"
                else:
                    # Multiple authors: use first author + "et al"
                    first_last, first_initial = parse_author_name(authors[0])
                    if first_initial:
                        human_id = f"[{first_last} {first_initial} et al. {year or 'Unknown'}]"
                    else:
                        human_id = f"[{first_last} et al. {year or 'Unknown'}]"
            else:
                # No authors available
                human_id = f"[Unknown {year or 'Unknown'}]"

            logger.debug(f"Generated human ID: {human_id}")

            # Extract clean publication info (remove authors, keep journal/publisher)
            publication = self._extract_publication_info(summary)

            # Create standardized paper dictionary according to PRD
            paper = {
                "paper_id": index + 1,  # Simple incrementing ID for search results
                "id": human_id,
                "title": title,
                "year": year,
                "publication": publication,
                "authors": authors if authors else [],
                "keywords": [],  # Empty for now, can be filled later
                "source": "google_scholar",
                "project_id": None,  # Will be set by search caller
                # Additional fields for compatibility
                "summary": snippet or summary,
                "link": link,
                "pdf_url": pdf_url,
                "publication_info": self._clean_publication_info(summary),
                "citation_count": citation_count,
                "cites_id": cites_id,
                "result_id": result_id,
                "metadata": {
                    "inline_links": inline_links,
                    "resources": resources,
                    "publication_info": pub_info
                }
            }

            # Try to get BibTeX link if available
            bibtex_link = self._get_bibtex_link(result_id)
            if bibtex_link:
                paper["bibtex_link"] = bibtex_link

            return paper

        except Exception as e:
            logger.error(f"Error extracting paper from result: {e}")
            return None



    def _extract_publication_info(self, summary: str) -> str:
        """
        Extract clean publication information from SerpAPI summary.
        Follows PRD format: "Journal Name, Volume(Issue), Pages"

        Args:
            summary: Raw publication summary from SerpAPI

        Returns:
            Clean publication info in PRD format
        """
        if not summary:
            return "Unknown Publication"

        # Remove author names (typically before the first dash)
        # Pattern: "Author1, Author2, ... - Journal Name, year - Publisher"
        parts = summary.split(' - ')

        if len(parts) >= 2:
            # Remove the first part (authors) and get journal info
            journal_part = parts[1].strip()

            # Clean up the journal name
            journal = journal_part.replace('…', '').strip()

            # Try to extract volume/issue/page info if present
            # Look for common patterns like "Journal Name, 2021, vol. 45(2), pp. 123-456"
            # or "Journal Name, vol. 45, issue 2"

            # For now, return the journal name as primary publication info
            # This matches the expected format better than including year/publisher
            return journal

        # Fallback: return the original summary if parsing fails
        return summary.strip()

    def _clean_publication_info(self, summary: str) -> str:
        """
        Clean publication info for the publication_info field.

        Args:
            summary: Raw publication summary from SerpAPI

        Returns:
            Cleaned publication info string
        """
        if not summary:
            return ""

        # For publication_info, we want to keep some author info but clean it up
        # Remove excessive ellipses and clean formatting
        cleaned = summary.replace('…', '').strip()

        # Remove multiple consecutive spaces
        while '  ' in cleaned:
            cleaned = cleaned.replace('  ', ' ')

        return cleaned



    def _find_pdf_url(self, resources: List[Dict]) -> Optional[str]:
        """
        Find PDF URL from resources list.

        Args:
            resources: List of resource dictionaries

        Returns:
            PDF URL or None
        """
        for resource in resources:
            if resource.get("file_format", "").upper() == "PDF":
                return resource.get("link")

        return None

    def _get_bibtex_link(self, result_id: str) -> Optional[str]:
        """
        Get BibTeX download link for a paper.

        Args:
            result_id: SerpAPI result ID

        Returns:
            BibTeX link or None
        """
        try:
            # This would require an additional API call to get BibTeX
            # For now, return None - can be implemented later
            return None

        except Exception as e:
            logger.debug(f"Could not get BibTeX link: {e}")
            return None

    def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific paper.

        Note: Google Scholar doesn't have persistent paper IDs,
        so this method may not work reliably for stored papers.

        Args:
            paper_id: Paper identifier (may not work for Google Scholar)

        Returns:
            Paper dictionary or None
        """
        # Google Scholar doesn't provide persistent IDs for papers
        # This would require re-searching or using title-based lookup
        logger.warning("Google Scholar engine doesn't support paper detail lookup by ID")
        return None

    def get_citing_papers(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Get papers that cite the specified paper.

        Args:
            paper_id: Paper identifier (cites_id from search results)

        Returns:
            List of citing paper dictionaries
        """
        try:
            if not paper_id:
                return []

            # Prepare citation search parameters
            params = {
                "engine": "google_scholar",
                "cites": paper_id,
                "api_key": self.api_key,
                "num": 10  # Limit results
            }

            logger.debug(f"Getting citing papers for ID: {paper_id}")

            response = requests.get(
                self.endpoint,
                params=params,
                timeout=self.config.get('request_timeout', 10.0)
            )
            response.raise_for_status()

            data = response.json()

            # Extract citing papers
            citing_papers = []
            organic_results = data.get("organic_results", [])

            for result in organic_results[:10]:  # Limit to 10 results
                paper = self._extract_paper_from_result(result, 0)
                if paper:
                    citing_papers.append(paper)

            logger.debug(f"Found {len(citing_papers)} citing papers")
            return citing_papers

        except requests.RequestException as e:
            logger.error(f"Citing papers API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting citing papers: {e}")
            return []

    def search_by_author(self, author_name: str, **kwargs) -> List[Dict[str, Any]]:
        """Search papers by author - not directly supported by Google Scholar API."""
        logger.info(f"Google Scholar author search not supported for: {author_name}")
        return [{"error": f"Author search not supported by {self.name}. Use general search with author name in query."}]

    def find_related_papers(self, reference_paper: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Find related papers - limited support through SerpAPI."""
        logger.info(f"Finding related papers for: {reference_paper.get('title', 'Unknown')}")
        return [{"error": f"Related papers discovery not fully supported by {self.name}. Use keyword-based search instead."}]

    def get_search_engine_categories(self) -> Dict[str, Any]:
        """Get Google Scholar categories - not applicable."""
        logger.info("Google Scholar categories not applicable")
        return {
            "engine_id": self.id,
            "engine_name": self.name,
            "categories": {},
            "total_categories": 0,
            "popular_categories": [],
            "usage_note": f"{self.name} doesn't use predefined categories. Use general search terms."
        }

    def analyze_paper_trends(self, papers: List[Dict[str, Any]], analysis_type: str) -> Dict[str, Any]:
        """Analyze paper trends - basic support."""
        if not papers:
            return {"error": "No papers to analyze"}

        try:
            if analysis_type == "authors":
                # Basic author analysis
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
                    "note": f"Basic analysis from {self.name} data"
                }

            elif analysis_type == "timeline":
                # Basic timeline analysis
                date_counts = {}
                for paper in papers:
                    date = paper.get("published_date", "")
                    if date:
                        year = date.split("-")[0]
                        date_counts[year] = date_counts.get(year, 0) + 1

                return {
                    "analysis_type": "timeline",
                    "papers_by_year": date_counts,
                    "note": f"Basic timeline analysis from {self.name} data"
                }

            else:
                return {"error": f"Analysis type '{analysis_type}' not supported by {self.name}"}

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def export_search_results(self, papers: List[Dict[str, Any]], format_type: str, **kwargs) -> Dict[str, Any]:
        """Export search results - basic support."""
        try:
            import json
            from datetime import datetime

            if not papers:
                return {"error": "No papers to export"}

            if format_type == "json":
                content = json.dumps(papers, indent=4)
            elif format_type == "bibtex":
                # Generate basic BibTeX from available data
                bibtex_entries = []
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

                    key = f"{first_author.lower()}{year}"

                    entry = f"""@article{{{key},
    title={{{title}}},
    author={{"{" and ".join(authors)}"}},
    year={{{year}}}
}}"""
                    bibtex_entries.append(entry)

                content = "\n\n".join(bibtex_entries)
            else:
                return {"error": f"Format '{format_type}' not supported by {self.name}"}

            return {
                "success": True,
                "format": format_type,
                "papers_exported": len(papers),
                "content_preview": content[:500] + ("..." if len(content) > 500 else ""),
                "export_data": content,
                "note": f"Basic export from {self.name} data"
            }

        except Exception as e:
            return {"error": f"Export failed: {str(e)}"}
