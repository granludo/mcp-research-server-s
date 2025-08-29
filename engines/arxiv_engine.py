"""
ArXiv Search Engine for Research MCP Server.

This module provides comprehensive ArXiv search capabilities including:
- Basic paper search with natural language processing
- Author-based search
- Related paper discovery using keyword similarity
- Trend analysis and export functionality
- Category-based filtering

Based on the excellent implementation from https://github.com/emi-dm/Arxiv-MCP
Adapted to fit our BaseSearchEngine interface.
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from collections import Counter

try:
    import arxiv
    from sklearn.feature_extraction.text import TfidfVectorizer
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    arxiv = None
    TfidfVectorizer = None

from search_engines import BaseSearchEngine

logger = logging.getLogger(__name__)


class ArXivEngine(BaseSearchEngine):
    """
    ArXiv search engine implementation.

    Provides comprehensive ArXiv search capabilities with advanced features
    like author search, related paper discovery, and trend analysis.
    """

    @property
    def name(self) -> str:
        """Human-readable name of the search engine."""
        return "ArXiv"

    @property
    def id(self) -> str:
        """Unique identifier for the search engine."""
        return "arxiv"

    def is_available(self) -> bool:
        """Check if this search engine is available."""
        if not ARXIV_AVAILABLE:
            logger.warning("python-arxiv library not installed")
            return False

        try:
            # Test basic connectivity by attempting a simple search
            search = arxiv.Search(query="test", max_results=1)
            next(search.results(), None)
            return True
        except Exception as e:
            logger.warning(f"ArXiv connectivity test failed: {e}")
            return False

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        stop_words = {
            "a", "an", "and", "the", "of", "in", "for", "to", "with", "on", "is", "are",
            "was", "were", "it", "this", "that", "these", "those", "paper", "research",
            "study", "method", "approach", "system", "model", "analysis", "results"
        }

        # Extract words and filter
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [
            word for word in words
            if len(word) > 2 and word not in stop_words
        ]

        return list(set(keywords))  # Remove duplicates

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple keyword-based similarity between two texts."""
        keywords1 = set(self._extract_keywords_from_text(text1))
        keywords2 = set(self._extract_keywords_from_text(text2))

        if not keywords1 or not keywords2:
            return 0.0

        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)

        return len(intersection) / len(union) if union else 0.0

    def _build_search_query(self, query: str, **kwargs) -> str:
        """Build ArXiv search query with advanced filtering."""
        query_parts = []

        # Handle natural language queries
        stop_words = {
            "a", "an", "and", "the", "of", "in", "for", "to", "with", "on", "is", "are"
        }

        # Extract years from query for date filtering
        years_in_query = re.findall(r'\b(20\d{2})\b', query)
        query_text = re.sub(r'\b(20\d{2})\b', "", query).strip()

        # Build keyword query
        keywords = [
            word for word in query_text.split()
            if word.lower() not in stop_words and len(word) > 2
        ]

        if keywords:
            keyword_query = " OR ".join([f'(ti:"{kw}" OR abs:"{kw}")' for kw in keywords])
            query_parts.append(f"({keyword_query})")
        else:
            query_parts.append(f'(ti:"{query_text}" OR abs:"{query_text}")')

        # Add category filter
        category = kwargs.get('category')
        if category:
            query_parts.append(f"cat:{category}")

        # Add date range
        date_from = kwargs.get('date_from')
        date_to = kwargs.get('date_to')

        # Use years from query if no explicit dates provided
        if not date_from and years_in_query:
            date_from = min(years_in_query)
        if not date_to and years_in_query:
            date_to = max(years_in_query)

        if date_from or date_to:
            start = "19910814"  # ArXiv founding date
            if date_from:
                try:
                    if len(date_from) == 4:  # YYYY
                        dt = datetime.strptime(date_from, "%Y")
                        start = dt.strftime("%Y%m%d")
                    else:  # YYYY-MM-DD
                        dt = datetime.strptime(date_from, "%Y-%m-%d")
                        start = dt.strftime("%Y%m%d")
                except ValueError:
                    logger.warning(f"Invalid date format: {date_from}")

            end = datetime.now().strftime("%Y%m%d")
            if date_to:
                try:
                    if len(date_to) == 4:  # YYYY
                        dt = datetime.strptime(date_to, "%Y")
                        dt = dt.replace(month=12, day=31)
                        end = dt.strftime("%Y%m%d")
                    else:  # YYYY-MM-DD
                        dt = datetime.strptime(date_to, "%Y-%m-%d")
                        end = dt.strftime("%Y%m%d")
                except ValueError:
                    logger.warning(f"Invalid date format: {date_to}")

            query_parts.append(f"submittedDate:[{start} TO {end}]")

        return " AND ".join(query_parts)

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for papers on ArXiv.

        Args:
            query: Search query (can be natural language)
            **kwargs: Additional search parameters (max_results, category, date_from, date_to)

        Returns:
            List of paper dictionaries with standardized format
        """
        if not self.is_available():
            logger.error("ArXiv engine is not available")
            return []

        try:
            max_results = kwargs.get('max_results', 10)
            sort_by_relevance = kwargs.get('sort_by_relevance', True)

            # Build the search query
            final_query = self._build_search_query(query, **kwargs)
            logger.debug(f"ArXiv query: {final_query}")

            # Configure search
            sort_criterion = (
                arxiv.SortCriterion.Relevance
                if sort_by_relevance
                else arxiv.SortCriterion.SubmittedDate
            )

            search = arxiv.Search(
                query=final_query,
                max_results=max_results,
                sort_by=sort_criterion,
                sort_order=arxiv.SortOrder.Descending,
            )

            results = []
            for r in search.results():
                # Generate human-readable ID
                authors = [a.name for a in r.authors]
                year = r.published.year
                if authors:
                    first_author = authors[0].split()[-1]
                    human_id = f"[{first_author} {year}]"
                else:
                    human_id = f"[Unknown {year}]"

                paper = {
                    "paper_id": len(results) + 1,  # Temporary ID for search results
                    "id": human_id,
                    "title": r.title,
                    "authors": authors,
                    "authors_string": ", ".join(authors),
                    "summary": r.summary,
                    "publication": f"arXiv:{r.entry_id.split('/')[-1]}",
                    "publication_year": year,
                    "source": "arxiv",
                    "source_url": r.entry_id,
                    "pdf_url": r.pdf_url,
                    "doi": getattr(r, 'doi', None),
                    "categories": getattr(r, 'categories', []),
                    "citation_count": getattr(r, 'citation_count', 0),
                    "keywords": [],  # ArXiv doesn't provide keywords, but we can extract them
                    "published_date": r.published.strftime("%Y-%m-%d"),
                    "updated_date": r.updated.strftime("%Y-%m-%d") if hasattr(r, 'updated') else None,
                    "arxiv_id": r.entry_id.split("/")[-1],
                    "project_id": kwargs.get('project_id'),
                }

                # Extract keywords from title and abstract
                text_content = f"{r.title} {r.summary}"
                paper["keywords"] = self._extract_keywords_from_text(text_content)[:10]  # Limit to 10 keywords

                results.append(paper)

            logger.info(f"Found {len(results)} papers from ArXiv")
            return results

        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []

    def search_by_author(self, author_name: str, **kwargs) -> List[Dict[str, Any]]:
        """Search papers by a specific author."""
        if not self.is_available():
            return []

        try:
            max_results = kwargs.get('max_results', 20)
            category = kwargs.get('category')
            date_from = kwargs.get('date_from')
            date_to = kwargs.get('date_to')

            query_parts = [f'au:"{author_name}"']

            if category:
                query_parts.append(f"cat:{category}")

            if date_from or date_to:
                start = "19910814"
                if date_from:
                    try:
                        if len(date_from) == 4:
                            dt = datetime.strptime(date_from, "%Y")
                            start = dt.strftime("%Y%m%d")
                        else:
                            dt = datetime.strptime(date_from, "%Y-%m-%d")
                            start = dt.strftime("%Y%m%d")
                    except ValueError:
                        pass

                end = datetime.now().strftime("%Y%m%d")
                if date_to:
                    try:
                        if len(date_to) == 4:
                            dt = datetime.strptime(date_to, "%Y")
                            dt = dt.replace(month=12, day=31)
                            end = dt.strftime("%Y%m%d")
                        else:
                            dt = datetime.strptime(date_to, "%Y-%m-%d")
                            end = dt.strftime("%Y%m%d")
                    except ValueError:
                        pass

                query_parts.append(f"submittedDate:[{start} TO {end}]")

            final_query = " AND ".join(query_parts)
            logger.debug(f"ArXiv author query: {final_query}")

            search = arxiv.Search(
                query=final_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            results = []
            for r in search.results():
                authors = [a.name for a in r.authors]
                year = r.published.year

                if authors:
                    first_author = authors[0].split()[-1]
                    human_id = f"[{first_author} {year}]"
                else:
                    human_id = f"[Unknown {year}]"

                paper = {
                    "paper_id": len(results) + 1,
                    "id": human_id,
                    "title": r.title,
                    "authors": authors,
                    "authors_string": ", ".join(authors),
                    "summary": r.summary,
                    "publication": f"arXiv:{r.entry_id.split('/')[-1]}",
                    "publication_year": year,
                    "source": "arxiv",
                    "source_url": r.entry_id,
                    "pdf_url": r.pdf_url,
                    "published_date": r.published.strftime("%Y-%m-%d"),
                    "arxiv_id": r.entry_id.split("/")[-1],
                    "categories": getattr(r, 'categories', []),
                    "keywords": self._extract_keywords_from_text(f"{r.title} {r.summary}")[:10],
                    "project_id": kwargs.get('project_id'),
                }
                results.append(paper)

            return results

        except Exception as e:
            logger.error(f"ArXiv author search failed: {e}")
            return []

    def find_related_papers(self, reference_paper: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Find papers related to a reference paper."""
        if not self.is_available():
            return []

        try:
            max_results = kwargs.get('max_results', 10)
            similarity_threshold = kwargs.get('similarity_threshold', 0.7)

            # Extract keywords from reference paper
            reference_text = f"{reference_paper.get('title', '')} {reference_paper.get('summary', '')}"
            reference_keywords = set(self._extract_keywords_from_text(reference_text))

            if not reference_keywords:
                return []

            # Build search query from reference keywords
            keyword_query = " OR ".join([f'(ti:"{kw}" OR abs:"{kw}")' for kw in list(reference_keywords)[:5]])
            search_query = f"({keyword_query})"

            # Exclude the reference paper if we have its ArXiv ID
            arxiv_id = reference_paper.get('arxiv_id')
            if arxiv_id:
                search_query += f" AND NOT (id:{arxiv_id})"

            search = arxiv.Search(
                query=search_query,
                max_results=max_results * 2,  # Get more to filter by similarity
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending,
            )

            results = []
            for r in search.results():
                # Calculate similarity
                paper_text = f"{r.title} {r.summary}"
                similarity = self._calculate_similarity(reference_text, paper_text)

                if similarity >= similarity_threshold:
                    authors = [a.name for a in r.authors]
                    year = r.published.year

                    if authors:
                        first_author = authors[0].split()[-1]
                        human_id = f"[{first_author} {year}]"
                    else:
                        human_id = f"[Unknown {year}]"

                    paper = {
                        "paper_id": len(results) + 1,
                        "id": human_id,
                        "title": r.title,
                        "authors": authors,
                        "authors_string": ", ".join(authors),
                        "summary": r.summary,
                        "publication": f"arXiv:{r.entry_id.split('/')[-1]}",
                        "publication_year": year,
                        "source": "arxiv",
                        "source_url": r.entry_id,
                        "pdf_url": r.pdf_url,
                        "published_date": r.published.strftime("%Y-%m-%d"),
                        "arxiv_id": r.entry_id.split("/")[-1],
                        "categories": getattr(r, 'categories', []),
                        "keywords": self._extract_keywords_from_text(paper_text)[:10],
                        "similarity_score": round(similarity, 3),
                        "project_id": kwargs.get('project_id'),
                    }
                    results.append(paper)

            # Sort by similarity and limit results
            results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            results = results[:max_results]

            return results

        except Exception as e:
            logger.error(f"ArXiv related papers search failed: {e}")
            return []

    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific paper by ArXiv ID."""
        if not self.is_available():
            return {"error": "ArXiv engine not available"}

        try:
            # Handle both ArXiv ID format and our internal ID format
            if paper_id.startswith("arxiv:") or "/" in paper_id:
                arxiv_id = paper_id.split("/")[-1].replace("arxiv:", "")
            else:
                # Try to extract ArXiv ID from the paper_id parameter
                arxiv_id = paper_id

            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())

            authors = [a.name for a in paper.authors]
            year = paper.published.year

            if authors:
                first_author = authors[0].split()[-1]
                human_id = f"[{first_author} {year}]"
            else:
                human_id = f"[Unknown {year}]"

            return {
                "paper_id": 1,  # Single result
                "id": human_id,
                "title": paper.title,
                "authors": authors,
                "authors_string": ", ".join(authors),
                "summary": paper.summary,
                "publication": f"arXiv:{paper.entry_id.split('/')[-1]}",
                "publication_year": year,
                "source": "arxiv",
                "source_url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "doi": getattr(paper, 'doi', None),
                "categories": getattr(paper, 'categories', []),
                "citation_count": getattr(paper, 'citation_count', 0),
                "published_date": paper.published.strftime("%Y-%m-%d"),
                "updated_date": paper.updated.strftime("%Y-%m-%d") if hasattr(paper, 'updated') else None,
                "arxiv_id": paper.entry_id.split("/")[-1],
                "journal_ref": getattr(paper, 'journal_ref', None),
                "comment": getattr(paper, 'comment', None),
                "primary_category": getattr(paper, 'primary_category', None),
                "keywords": self._extract_keywords_from_text(f"{paper.title} {paper.summary}")[:10],
            }

        except Exception as e:
            logger.error(f"Failed to fetch ArXiv paper details: {e}")
            return {"error": f"Failed to fetch paper details: {str(e)}"}

    def get_citing_papers(self, paper_id: str) -> List[Dict[str, Any]]:
        """Get papers that cite the given paper (ArXiv doesn't provide this directly)."""
        # ArXiv doesn't provide citation data directly through their API
        # This would require integration with Semantic Scholar or other citation databases
        logger.info("ArXiv citation data not available through API")
        return []

    def get_search_engine_categories(self) -> Dict[str, Any]:
        """Get available ArXiv categories for search engine-agnostic interface."""
        categories = {
            "Computer Science": {
                "cs.AI": "Artificial Intelligence",
                "cs.AR": "Hardware Architecture",
                "cs.CC": "Computational Complexity",
                "cs.CE": "Computational Engineering, Finance, and Science",
                "cs.CG": "Computational Geometry",
                "cs.CL": "Computation and Language",
                "cs.CR": "Cryptography and Security",
                "cs.CV": "Computer Vision and Pattern Recognition",
                "cs.CY": "Computers and Society",
                "cs.DB": "Databases",
                "cs.DC": "Distributed, Parallel, and Cluster Computing",
                "cs.DL": "Digital Libraries",
                "cs.DM": "Discrete Mathematics",
                "cs.DS": "Data Structures and Algorithms",
                "cs.ET": "Emerging Technologies",
                "cs.FL": "Formal Languages and Automata Theory",
                "cs.GL": "General Literature",
                "cs.GR": "Graphics",
                "cs.GT": "Computer Science and Game Theory",
                "cs.HC": "Human-Computer Interaction",
                "cs.IR": "Information Retrieval",
                "cs.IT": "Information Theory",
                "cs.LG": "Machine Learning",
                "cs.LO": "Logic in Computer Science",
                "cs.MA": "Multiagent Systems",
                "cs.MM": "Multimedia",
                "cs.MS": "Mathematical Software",
                "cs.NA": "Numerical Analysis",
                "cs.NE": "Neural and Evolutionary Computing",
                "cs.NI": "Networking and Internet Architecture",
                "cs.OH": "Other Computer Science",
                "cs.OS": "Operating Systems",
                "cs.PF": "Performance",
                "cs.PL": "Programming Languages",
                "cs.RO": "Robotics",
                "cs.SC": "Symbolic Computation",
                "cs.SD": "Sound",
                "cs.SE": "Software Engineering",
                "cs.SI": "Social and Information Networks",
                "cs.SY": "Systems and Control"
            }
        }

        return {
            "engine_id": self.id,
            "engine_name": self.name,
            "categories": categories,
            "total_categories": sum(len(cats) for cats in categories.values()),
            "popular_categories": ["cs.LG", "cs.AI", "cs.CV", "cs.CL", "cs.SE", "cs.RO"],
            "usage_note": f"Use category codes (e.g., 'cs.AI') in {self.name} search functions"
        }

    def analyze_paper_trends(self, papers: List[Dict[str, Any]], analysis_type: str) -> Dict[str, Any]:
        """Analyze trends in a collection of papers."""
        if not papers:
            return {"error": "No papers to analyze"}

        try:
            if analysis_type == "authors":
                author_counts = Counter()
                for paper in papers:
                    authors = paper.get("authors", [])
                    if isinstance(authors, str):
                        authors = [authors]
                    for author in authors:
                        if isinstance(author, str):
                            author_counts[author] += 1

                return {
                    "analysis_type": "authors",
                    "total_unique_authors": len(author_counts),
                    "most_prolific_authors": author_counts.most_common(10),
                    "collaboration_stats": {
                        "avg_authors_per_paper": sum(len(p.get("authors", [])) for p in papers) / len(papers),
                        "single_author_papers": sum(1 for p in papers if len(p.get("authors", [])) == 1),
                        "multi_author_papers": sum(1 for p in papers if len(p.get("authors", [])) > 1),
                    }
                }

            elif analysis_type == "timeline":
                date_counts = Counter()
                for paper in papers:
                    date = paper.get("published_date", "")
                    if date:
                        year = date.split("-")[0]
                        date_counts[year] += 1

                return {
                    "analysis_type": "timeline",
                    "papers_by_year": dict(sorted(date_counts.items())),
                    "most_active_year": date_counts.most_common(1)[0] if date_counts else None,
                    "total_years_span": len(date_counts),
                }

            elif analysis_type == "categories":
                category_counts = Counter()
                for paper in papers:
                    categories = paper.get("categories", [])
                    for cat in categories:
                        category_counts[cat] += 1

                return {
                    "analysis_type": "categories",
                    "total_categories": len(category_counts),
                    "most_common_categories": category_counts.most_common(10),
                    "category_distribution": dict(category_counts),
                }

            elif analysis_type == "keywords":
                # Extract keywords from titles and abstracts
                text_content = []
                for paper in papers:
                    title = paper.get("title", "")
                    summary = paper.get("summary", "")
                    text_content.append(f"{title} {summary}")

                if text_content and TfidfVectorizer:
                    try:
                        vectorizer = TfidfVectorizer(
                            max_features=50,
                            stop_words='english',
                            ngram_range=(1, 2),
                            min_df=2
                        )
                        tfidf_matrix = vectorizer.fit_transform(text_content)
                        feature_names = vectorizer.get_feature_names_out()
                        scores = tfidf_matrix.sum(axis=0).A1

                        keyword_scores = list(zip(feature_names, scores))
                        keyword_scores.sort(key=lambda x: x[1], reverse=True)

                        return {
                            "analysis_type": "keywords",
                            "top_keywords": keyword_scores[:20],
                            "total_unique_terms": len(feature_names),
                        }
                    except Exception as e:
                        return {
                            "analysis_type": "keywords",
                            "error": f"Could not perform keyword analysis: {str(e)}",
                        }
                else:
                    return {
                        "analysis_type": "keywords",
                        "error": "Keyword analysis not available (missing sklearn)",
                    }

            else:
                return {"error": f"Unsupported analysis type: {analysis_type}"}

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def export_search_results(self, papers: List[Dict[str, Any]], format_type: str, **kwargs) -> Dict[str, Any]:
        """Export search results to various formats."""
        try:
            import os
            import json
            import pandas as pd
            from datetime import datetime

            if not papers:
                return {"error": "No papers to export"}

            # Generate default filename if not provided
            filename = kwargs.get('filename', f"arxiv_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            if format_type == "bibtex":
                bibtex_entries = []
                export_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                header = f"""% ArXiv Export from {self.name}
% Exported: {export_time}
"""
                bibtex_entries.append(header)

                bibtex_keys = set()
                for i, paper in enumerate(papers):
                    authors = paper.get("authors", ["unknown"])
                    year = paper.get("published_date", "unknown").split("-")[0]

                    if isinstance(authors, str):
                        authors = [authors]

                    first_author_lastname = "unknown"
                    if authors and authors[0] != "unknown":
                        name_parts = authors[0].split(" ")
                        if name_parts:
                            first_author_lastname = name_parts[-1]

                    first_author_lastname = re.sub(r'[^a-zA-Z0-9]', '', first_author_lastname).lower()

                    key = f"{first_author_lastname}{year}"

                    # Handle duplicates
                    original_key = key
                    suffix = 1
                    while key in bibtex_keys:
                        key = f"{original_key}_{suffix}"
                        suffix += 1
                    bibtex_keys.add(key)

                    title = paper.get("title", "No Title Provided")
                    author_str = " and ".join(authors)
                    pdf_url = paper.get("pdf_url", "")

                    arxiv_id = paper.get("arxiv_id", "")
                    if arxiv_id:
                        journal = f"arXiv preprint arXiv:{arxiv_id}"
                    else:
                        journal = f"arXiv preprint arXiv:{key}"

                    entry = f"""@article{{{key},
    title = {{{title}}},
    author = {{{author_str}}},
    year = {{{year}}},
    journal = {{{journal}}},
    url = {{{pdf_url}}}
}}"""
                    bibtex_entries.append(entry)

                content = "\n\n".join(bibtex_entries)

            elif format_type == "csv":
                df = pd.DataFrame(papers)
                content = df.to_string()
                # Note: In real implementation, this would save to file

            elif format_type == "json":
                content = json.dumps(papers, indent=4)

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
                "export_data": content  # In real implementation, this would be saved to file
            }

        except Exception as e:
            return {"error": f"Export failed: {str(e)}"}
