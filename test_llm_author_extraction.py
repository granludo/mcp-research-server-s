#!/usr/bin/env python3
"""
Focused test for LLM-based author extraction.
"""

import os
import sys
sys.path.append('.')

from search_engines import BaseSearchEngine

class TestEngine(BaseSearchEngine):
    @property
    def name(self): return "Test Engine"
    @property
    def id(self): return "test"
    def is_available(self): return True
    def search(self, query, **kwargs): return []
    def get_paper_details(self, paper_id): return None

def test_llm_author_extraction():
    """Test LLM-based author extraction specifically."""

    print('🤖 Testing LLM-Based Author Extraction')
    print('=' * 50)

    engine = TestEngine()

    # Check if OpenAI is available and API key exists
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    print(f'OpenAI API Key available: {has_api_key}')

    # Test with the exact case from user's example
    test_pub_info = {
        "summary": "TC Ma, DE Willis - Frontiers in molecular neuroscience, 2015 - frontiersin.org"
    }

    print(f'\nInput: {test_pub_info["summary"]}')

    # Test the LLM extraction directly
    authors_llm = engine._extract_authors_with_llm(test_pub_info)
    print(f'LLM extraction: {authors_llm}')

    # Test the regex fallback
    authors_regex = engine._extract_authors_from_text(test_pub_info)
    print(f'Regex extraction: {authors_regex}')

    # Test the full centralized extraction
    authors_full = engine._extract_authors(test_pub_info)
    print(f'Full extraction: {authors_full}')

    # Test different cases
    test_cases = [
        "Smith, J., Johnson, A. - Journal of Science, 2020",
        "T.C. Ma, David E. Willis - Frontiers, 2015",
        "Complex, Name O., Another, Author B. - Journal, 2023",
        "Single Author - Journal, 2019"
    ]

    print('\n📋 Testing Different Author Formats:')
    print('-' * 40)

    for i, case in enumerate(test_cases, 1):
        print(f'\nTest {i}: {case}')
        test_info = {"summary": case}
        authors = engine._extract_authors(test_info)
        print(f'  Result: {authors}')

    print('\n✅ LLM Author Extraction Test Complete!')

    # Summary
    success_count = sum(1 for case in test_cases
                       if len(engine._extract_authors({"summary": case})) > 0)
    print(f'\n📊 Summary: {success_count}/{len(test_cases)} test cases extracted authors')

if __name__ == '__main__':
    test_llm_author_extraction()
