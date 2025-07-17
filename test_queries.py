#!/usr/bin/env python3

import sys
import os
import json
from src.query_engine import QueryEngine


def test_assignment_queries():
    """Test the specific queries from the assignment."""

    # Initialize query engine
    query_engine = QueryEngine(
        embedding_model="all-MiniLM-L6-v2",
        llm_model="distilgpt2",
        chunk_size=400,
        overlap=50
    )

    # Setup from documents
    print("Setting up query engine...")
    success = query_engine.setup_from_documents("data")

    if not success:
        print("Failed to setup query engine. Please ensure you have documents in the data/ directory.")
        return

    # Assignment test queries
    test_queries = [
        {
            "id": 1,
            "query": "Give me the correct coded classification for the following diagnosis: Recurrent depressive disorder, currently in remission",
            "expected_keywords": ["F33.4", "ICD-10", "remission"]
        },
        {
            "id": 2,
            "query": "What are the diagnostic criteria for Obsessive-Compulsive Disorder (OCD)?",
            "expected_keywords": ["DSM-5", "obsessions", "compulsions", "criteria"]
        }
    ]

    print("\n" + "=" * 80)
    print("ASSIGNMENT QUERY TESTING")
    print("=" * 80)

    results = []

    for test_case in test_queries:
        print(f"\nTEST CASE {test_case['id']}")
        print(f"Query: {test_case['query']}")
        print("-" * 60)

        # Process query
        result = query_engine.query(test_case['query'], top_k=5, verbose=True)

        # Display results
        print(f"\nGenerated Answer:")
        print(f"{result['answer']}")

        print(f"\nProcessing Time: {result['processing_time']:.2f} seconds")

        # Check for expected keywords
        answer_lower = result['answer'].lower()
        found_keywords = [kw for kw in test_case['expected_keywords'] if kw.lower() in answer_lower]

        print(f"\nExpected Keywords: {test_case['expected_keywords']}")
        print(f"Found Keywords: {found_keywords}")
        print(f"Keyword Match Rate: {len(found_keywords)}/{len(test_case['expected_keywords'])}")

        # Display retrieved chunks
        if 'retrieved_chunks' in result:
            print(f"\nRetrieved Chunks ({len(result['retrieved_chunks'])}):")
            for i, chunk in enumerate(result['retrieved_chunks'][:3]):  # Show top 3
                print(f"  {i + 1}. Score: {chunk['similarity_score']:.3f}")
                print(f"     Source: {chunk['filename']}")
                print(f"     Content: {chunk['content'][:150]}...")
                print()

        # Store result for summary
        results.append({
            'test_id': test_case['id'],
            'query': test_case['query'],
            'answer': result['answer'],
            'processing_time': result['processing_time'],
            'keyword_match_rate': len(found_keywords) / len(test_case['expected_keywords']),
            'retrieved_chunks_count': len(result.get('retrieved_chunks', []))
        })

        print("=" * 80)

    # Summary
    print("\nTEST SUMMARY")
    print("=" * 40)

    total_time = sum(r['processing_time'] for r in results)
    avg_keyword_match = sum(r['keyword_match_rate'] for r in results) / len(results)

    print(f"Total Tests: {len(results)}")
    print(f"Total Processing Time: {total_time:.2f} seconds")
    print(f"Average Processing Time: {total_time / len(results):.2f} seconds")
    print(f"Average Keyword Match Rate: {avg_keyword_match:.2%}")

    for result in results:
        print(f"\nTest {result['test_id']}:")
        print(f"  Processing Time: {result['processing_time']:.2f}s")
        print(f"  Keyword Match: {result['keyword_match_rate']:.2%}")
        print(f"  Retrieved Chunks: {result['retrieved_chunks_count']}")

    return results


def test_additional_queries():
    """Test additional queries to verify system robustness."""

    query_engine = QueryEngine()

    if not query_engine.setup_from_documents("data"):
        print("Failed to setup query engine.")
        return

    additional_queries = [
        "What is the difference between obsessions and compulsions?",
        "How long must symptoms be present for OCD diagnosis?",
        "What does F33.4 mean in medical coding?",
        "What are the treatment options for depression?",
        "How is remission defined in depression?"
    ]

    print("\n" + "=" * 80)
    print("ADDITIONAL QUERY TESTING")
    print("=" * 80)

    for i, query in enumerate(additional_queries, 1):
        print(f"\nAdditional Query {i}: {query}")
        print("-" * 50)

        result = query_engine.query(query, top_k=3, verbose=False)

        print(f"Answer: {result['answer']}")
        print(f"Time: {result['processing_time']:.2f}s")

        if result.get('retrieved_chunks'):
            best_chunk = result['retrieved_chunks'][0]
            print(f"Best match (score: {best_chunk['similarity_score']:.3f}): {best_chunk['content'][:100]}...")


def main():
    """Main function to run all tests."""

    # Check if data directory exists
    if not os.path.exists("data"):
        print("Error: data/ directory not found.")
        print("Please run main.py first to create sample data, or add your own documents to data/")
        return

    if not os.listdir("data"):
        print("Error: data/ directory is empty.")
        print("Please run main.py first to create sample data, or add your own documents to data/")
        return

    print("RAG QA System - Test Suite")
    print("=" * 40)

    # Run assignment tests
    assignment_results = test_assignment_queries()

    # Run additional tests
    test_additional_queries()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()