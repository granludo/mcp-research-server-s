import arxiv
import argparse
import json
import os
import requests
import sys
from typing import List
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Global verbose flag (set temporarily to capture initial debug info)
VERBOSE = True

def debug_print(message: str):
    """Print debug message to stderr if verbose mode is enabled."""
    if VERBOSE:
        print(f"[DEBUG] {message}", file=sys.stderr)

# Load environment variables from .env file
debug_print(f"Current working directory: {os.getcwd()}")
debug_print(f"Looking for .env file at: {os.path.join(os.getcwd(), '.env')}")
debug_print(f".env file exists: {os.path.exists('.env')}")

load_dotenv()

PAPER_DIR = "papers"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Research Scholar MCP Server')
parser.add_argument('--verbose', action='store_true', help='Enable verbose debugging output')
args = parser.parse_args()

# Set VERBOSE based on command line argument, but keep initial debug messages
user_wants_verbose = args.verbose
if not user_wants_verbose:
    VERBOSE = False

# Test the debug function immediately
debug_print("Verbose mode enabled")
debug_print(f"Environment loaded. SERP_API_KEY present: {'SERP_API_KEY' in os.environ}")
if 'SERP_API_KEY' in os.environ:
    debug_print(f"SERP_API_KEY length: {len(os.environ['SERP_API_KEY'])} characters")

# Reset VERBOSE to user preference
VERBOSE = user_wants_verbose

# Initialize FastMCP server
mcp = FastMCP("research-scholar")

@mcp.tool()
def search_papers(topic: str, source: str = "all-sources", max_results: int = 5) -> List[str]:
    """
    Search for papers on Google Scholar, ArXiv, or both based on a topic and store their information.

    Args:
        topic: The topic to search for
        source: The source to search in ("google-scholar", "arxiv", or "all-sources") (default: "all-sources")
        max_results: Maximum number of results to retrieve per source (default: 5)

    Returns:
        List of paper IDs found in the search
    """
    debug_print(f"Starting search_papers with topic='{topic}', source='{source}', max_results={max_results}")

    if source not in ["google-scholar", "arxiv", "all-sources"]:
        debug_print(f"Invalid source parameter: {source}")
        raise ValueError("Source must be one of: 'google-scholar', 'arxiv', or 'all-sources'")

    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    debug_print(f"Creating directory: {path}")
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "papers_info.json")
    debug_print(f"Papers will be saved to: {file_path}")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
        debug_print(f"Loaded existing papers_info with {len(papers_info)} entries")
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}
        debug_print("No existing papers_info file found, starting fresh")

    paper_ids = []
    search_errors = []

    # Search Google Scholar if requested
    if source in ["google-scholar", "all-sources"]:
        debug_print("Searching Google Scholar...")
        try:
            scholar_results = search_google_scholar(topic, max_results)
            debug_print(f"Google Scholar returned {len(scholar_results)} results")
            for result in scholar_results:
                paper_id = result.get('paper_id', f"scholar_{len(paper_ids)}")
                paper_ids.append(paper_id)
                papers_info[paper_id] = result
        except (ValueError, ConnectionError, RuntimeError) as e:
            error_msg = f"Google Scholar search failed: {str(e)}"
            debug_print(error_msg)
            search_errors.append(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during Google Scholar search: {str(e)}"
            debug_print(error_msg)
            search_errors.append(error_msg)

    # Search ArXiv if requested
    if source in ["arxiv", "all-sources"]:
        debug_print("Searching ArXiv...")
        try:
            arxiv_results = search_arxiv(topic, max_results)
            debug_print(f"ArXiv returned {len(arxiv_results)} results")
            for paper in arxiv_results:
                paper_id = paper.get_short_id()
                paper_ids.append(paper_id)
                paper_info = {
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'summary': paper.summary,
                    'pdf_url': paper.pdf_url,
                    'published': str(paper.published.date())
                }
                papers_info[paper_id] = paper_info
        except (ConnectionError, RuntimeError) as e:
            error_msg = f"ArXiv search failed: {str(e)}"
            debug_print(error_msg)
            search_errors.append(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during ArXiv search: {str(e)}"
            debug_print(error_msg)
            search_errors.append(error_msg)

    # If no results found and we have errors, raise an exception with all error details
    if not paper_ids and search_errors:
        all_errors = "; ".join(search_errors)
        raise RuntimeError(f"No papers could be retrieved due to the following errors: {all_errors}")

    # If no results found but no errors occurred (shouldn't happen in normal cases)
    if not paper_ids:
        raise RuntimeError(f"No papers found for topic '{topic}' using source '{source}'. This may indicate an issue with the search query or API services.")

    # Save updated papers_info to json file
    debug_print(f"Saving {len(papers_info)} papers to {file_path}")
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)

    print(f"Results are saved in: {file_path}")
    debug_print(f"Search completed. Total papers found: {len(paper_ids)}")

    return paper_ids


def search_google_scholar(topic: str, max_results: int = 5) -> List[dict]:
    """
    Search Google Scholar using SERP API.

    Args:
        topic: The search topic
        max_results: Maximum number of results to retrieve

    Returns:
        List of paper dictionaries from Google Scholar

    Raises:
        ValueError: If SERP_API_KEY is not set
        ConnectionError: If the API request fails due to network issues
        RuntimeError: If the API returns an error response
    """
    debug_print(f"Starting Google Scholar search for topic: '{topic}' with max_results: {max_results}")

    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        error_msg = "SERP_API_KEY environment variable not set. Please set your SerpAPI key to search Google Scholar."
        debug_print(error_msg)
        raise ValueError(error_msg)
    else:
        debug_print("SERP_API_KEY found in environment variables")

    params = {
        "engine": "google_scholar",
        "q": topic,
        "api_key": api_key,
        "num": max_results
    }

    debug_print(f"SerpAPI request parameters: engine={params['engine']}, q='{params['q']}', num={params['num']}")
    debug_print("Making request to https://serpapi.com/search")

    try:
        response = requests.get("https://serpapi.com/search", params=params)
        debug_print(f"SerpAPI response status code: {response.status_code}")

        response.raise_for_status()
        results = response.json()

        # Check for API-specific errors in the response
        if "error" in results:
            error_msg = f"SerpAPI error: {results['error']}"
            debug_print(error_msg)
            raise RuntimeError(error_msg)

        debug_print(f"SerpAPI response contains {len(results.get('organic_results', []))} organic results")

        papers = []
        for i, article in enumerate(results.get("organic_results", [])):
            paper_info = {
                'title': article.get("title", "Unknown Title"),
                'authors': article.get("authors", []),
                'summary': article.get("snippet", "No summary available"),
                'link': article.get("link", ""),
                'published': article.get("publication_info", {}).get("summary", "Unknown date"),
                'source': 'google_scholar',
                'paper_id': f"scholar_{i}"
            }
            papers.append(paper_info)
            debug_print(f"Processed paper {i+1}: '{paper_info['title'][:50]}...' from {len(paper_info['authors'])} authors")

        debug_print(f"Successfully retrieved {len(papers)} papers from Google Scholar")
        return papers

    except requests.ConnectionError as e:
        error_msg = f"Network connection error while accessing SerpAPI: {str(e)}. Please check your internet connection."
        debug_print(error_msg)
        raise ConnectionError(error_msg)
    except requests.Timeout as e:
        error_msg = f"Request timeout while accessing SerpAPI: {str(e)}. The service may be temporarily unavailable."
        debug_print(error_msg)
        raise ConnectionError(error_msg)
    except requests.HTTPError as e:
        error_msg = f"HTTP error while accessing SerpAPI (status {response.status_code}): {str(e)}"
        debug_print(error_msg)
        raise RuntimeError(error_msg)
    except requests.RequestException as e:
        error_msg = f"Request error while accessing SerpAPI: {str(e)}"
        debug_print(error_msg)
        raise RuntimeError(error_msg)
    except ValueError as e:
        error_msg = f"Invalid response format from SerpAPI: {str(e)}"
        debug_print(error_msg)
        raise RuntimeError(error_msg)


def search_arxiv(topic: str, max_results: int = 5) -> List:
    """
    Search ArXiv for papers.

    Args:
        topic: The search topic
        max_results: Maximum number of results to retrieve

    Returns:
        List of ArXiv paper objects

    Raises:
        ConnectionError: If the ArXiv API request fails due to network issues
        RuntimeError: If the ArXiv API returns an error or invalid response
    """
    debug_print(f"Starting ArXiv search for topic: '{topic}' with max_results: {max_results}")

    debug_print("Initializing ArXiv client")
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    debug_print(f"ArXiv search parameters: query='{search.query}', max_results={search.max_results}, sort_by={search.sort_by}")
    debug_print("Making request to ArXiv API")

    try:
        results = list(client.results(search))
        debug_print(f"ArXiv API returned {len(results)} results")

        for i, paper in enumerate(results):
            debug_print(f"Processed ArXiv paper {i+1}: '{paper.title[:50]}...' by {len(paper.authors)} authors")

        debug_print(f"Successfully retrieved {len(results)} papers from ArXiv")
        return results

    except Exception as e:
        error_msg = f"ArXiv search failed: {str(e)}. Please check your internet connection and try again."
        debug_print(error_msg)
        raise ConnectionError(error_msg)

@mcp.tool()
def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.
    
    Args:
        paper_id: The ID of the paper to look for
        
    Returns:
        JSON string with paper information if found, error message if not found
    """
 
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
    
    return f"There's no saved information related to paper {paper_id}."



@mcp.resource("papers://folders")
def get_available_folders() -> str:
    """
    List all available topic folders in the papers directory.
    
    This resource provides a simple list of all available topic folders.
    """
    folders = []
    
    # Get all topic directories
    if os.path.exists(PAPER_DIR):
        for topic_dir in os.listdir(PAPER_DIR):
            topic_path = os.path.join(PAPER_DIR, topic_dir)
            if os.path.isdir(topic_path):
                papers_file = os.path.join(topic_path, "papers_info.json")
                if os.path.exists(papers_file):
                    folders.append(topic_dir)
    
    # Create a simple markdown list
    content = "# Available Topics\n\n"
    if folders:
        for folder in folders:
            content += f"- {folder}\n"
        content += f"\nUse @{folder} to access papers in that topic.\n"
    else:
        content += "No topics found.\n"
    
    return content

@mcp.resource("papers://{topic}")
def get_topic_papers(topic: str) -> str:
    """
    Get detailed information about papers on a specific topic.
    
    Args:
        topic: The research topic to retrieve papers for
    """
    topic_dir = topic.lower().replace(" ", "_")
    papers_file = os.path.join(PAPER_DIR, topic_dir, "papers_info.json")
    
    if not os.path.exists(papers_file):
        return f"# No papers found for topic: {topic}\n\nTry searching for papers on this topic first."
    
    try:
        with open(papers_file, 'r') as f:
            papers_data = json.load(f)
        
        # Create markdown content with paper details
        content = f"# Papers on {topic.replace('_', ' ').title()}\n\n"
        content += f"Total papers: {len(papers_data)}\n\n"
        
        for paper_id, paper_info in papers_data.items():
            content += f"## {paper_info['title']}\n"
            content += f"- **Paper ID**: {paper_id}\n"
            content += f"- **Authors**: {', '.join(paper_info['authors']) if isinstance(paper_info['authors'], list) else paper_info['authors']}\n"
            content += f"- **Published**: {paper_info['published']}\n"
            if 'source' in paper_info:
                content += f"- **Source**: {paper_info['source']}\n"
            if 'pdf_url' in paper_info:
                content += f"- **PDF URL**: [{paper_info['pdf_url']}]({paper_info['pdf_url']})\n"
            elif 'link' in paper_info:
                content += f"- **Link**: [{paper_info['link']}]({paper_info['link']})\n"
            content += "\n"
            content += f"### Summary\n{paper_info['summary'][:500]}...\n\n"
            content += "---\n\n"
        
        return content
    except json.JSONDecodeError:
        return f"# Error reading papers data for {topic}\n\nThe papers data file is corrupted."

@mcp.prompt()
def generate_search_prompt(topic: str, source: str = "all-sources", num_papers: int = 5) -> str:
    """Generate a prompt for Claude to find and discuss academic papers on a specific topic from specified sources."""
    return f"""Search for {num_papers} academic papers about '{topic}' using the search_papers tool with source='{source}'.

Follow these instructions:
1. First, search for papers using search_papers(topic='{topic}', source='{source}', max_results={num_papers})
2. For each paper found, extract and organize the following information:
   - Paper title
   - Authors
   - Publication date
   - Brief summary of the key findings
   - Main contributions or innovations
   - Methodologies used
   - Relevance to the topic '{topic}'
   - Source (Google Scholar, ArXiv, etc.)

3. Provide a comprehensive summary that includes:
   - Overview of the current state of research in '{topic}'
   - Common themes and trends across the papers
   - Key research gaps or areas for future investigation
   - Most impactful or influential papers in this area
   - Comparison between different sources if multiple sources were used

4. Organize your findings in a clear, structured format with headings and bullet points for easy readability.

Available sources:
- 'google-scholar': Search only Google Scholar
- 'arxiv': Search only ArXiv
- 'all-sources': Search both Google Scholar and ArXiv (default)

Please present both detailed information about each paper and a high-level synthesis of the research landscape in {topic}."""

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')