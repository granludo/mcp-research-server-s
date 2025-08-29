#!/usr/bin/env python3
"""
Test script to verify deduplication functionality.
"""

import sys
import os
import tempfile
import asyncio
sys.path.append('.')

from database import DatabaseManager

async def test_deduplication():
    """Test the deduplication functionality."""

    print('🧪 Testing Deduplication System')
    print('=' * 50)

    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_manager = DatabaseManager(Path(temp_dir) / "test.db")
        await db_manager.initialize()

        # Test paper data
        test_paper_1 = {
            'title': 'Test Paper One',
            'authors': ['Smith, J.', 'Doe, A.'],
            'year': 2023,
            'source': 'test',
            'link': 'https://example.com/paper1'
        }

        test_paper_2 = {
            'title': 'Test Paper Two',
            'authors': ['Johnson, B.'],
            'year': 2024,
            'source': 'test',
            'doi': '10.1000/test.doi'
        }

        # Duplicate of paper 1 (same title, authors, year)
        duplicate_paper_1 = {
            'title': 'Test Paper One',  # Same title
            'authors': ['Smith, J.', 'Doe, A.'],  # Same authors
            'year': 2023,  # Same year
            'source': 'different_source',  # Different source
            'link': 'https://different.com/paper1'  # Different URL
        }

        # Similar paper (different title but same DOI)
        similar_paper_2 = {
            'title': 'Different Title for Same Paper',  # Different title
            'authors': ['Different, A.'],  # Different authors
            'year': 2025,  # Different year
            'source': 'test',
            'doi': '10.1000/test.doi'  # Same DOI
        }

        print('\n📝 Storing initial papers...')
        project_id = "dedup_test"

        # Store initial papers
        initial_results = db_manager.store_search_results([test_paper_1, test_paper_2], project_id)
        print(f'Initial storage: {len(initial_results)} papers')
        for paper in initial_results:
            status = "NEW" if paper.get('is_new_paper') else "EXISTING"
            print(f'  - {paper["title"]}: {status} (ID: {paper["paper_id"]})')

        print('\n🔍 Testing deduplication...')

        # Try to store duplicates
        duplicate_results = db_manager.store_search_results([duplicate_paper_1, similar_paper_2], project_id)
        print(f'Duplicate storage: {len(duplicate_results)} papers')
        for paper in duplicate_results:
            status = "NEW" if paper.get('is_new_paper') else "EXISTING"
            print(f'  - {paper["title"]}: {status} (ID: {paper["paper_id"]})')

        # Verify deduplication worked
        new_papers = sum(1 for paper in duplicate_results if paper.get('is_new_paper', False))
        existing_papers = sum(1 for paper in duplicate_results if not paper.get('is_new_paper', True))

        print(f'\n📊 Deduplication Results:')
        print(f'  New papers: {new_papers}')
        print(f'  Existing papers: {existing_papers}')

        # Test the find_duplicate_paper method directly
        print('\n🔬 Testing duplicate detection...')

        # This should find the existing paper
        duplicate = db_manager.find_duplicate_paper(duplicate_paper_1)
        if duplicate:
            print(f'✅ Found duplicate for paper 1: {duplicate["title"]} (ID: {duplicate["paper_id"]})')
        else:
            print('❌ Failed to find duplicate for paper 1')

        # This should find the existing paper by DOI
        duplicate = db_manager.find_duplicate_paper(similar_paper_2)
        if duplicate:
            print(f'✅ Found duplicate for paper 2 (by DOI): {duplicate["title"]} (ID: {duplicate["paper_id"]})')
        else:
            print('❌ Failed to find duplicate for paper 2')

        # This should NOT find a duplicate (completely new paper)
        new_paper = {
            'title': 'Completely New Paper',
            'authors': ['New, A.'],
            'year': 2025,
            'source': 'test'
        }
        duplicate = db_manager.find_duplicate_paper(new_paper)
        if not duplicate:
            print('✅ Correctly identified new paper (no duplicate found)')
        else:
            print('❌ Incorrectly found duplicate for new paper')

        print('\n✅ Deduplication test completed!')

        # Summary
        total_papers = len(initial_results) + len(duplicate_results)
        unique_papers = len(set(paper['paper_id'] for paper in initial_results + duplicate_results))

        print(f'\n📈 Summary:')
        print(f'  Total papers processed: {total_papers}')
        print(f'  Unique papers in database: {unique_papers}')
        print(f'  Duplicates prevented: {total_papers - unique_papers}')

        success = unique_papers == 2  # Should have exactly 2 unique papers
        if success:
            print('🎉 Deduplication working correctly!')
        else:
            print('❌ Deduplication may have issues')

        return success

if __name__ == '__main__':
    from pathlib import Path
    asyncio.run(test_deduplication())
