#!/usr/bin/env python3

import os
import sys
import json
from src.query_engine import QueryEngine


def create_sample_data():
    """Create sample medical/clinical data for testing."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Sample clinical guidelines content
    sample_content = """
    Diagnostic Criteria for Obsessive-Compulsive Disorder (OCD)

    According to the DSM-5, the diagnostic criteria for OCD include:

    A. Presence of obsessions, compulsions, or both:

    Obsessions are defined by:
    1. Recurrent and persistent thoughts, urges, or images that are experienced as intrusive and unwanted
    2. The individual attempts to ignore or suppress such thoughts, urges, or images, or to neutralize them with some other thought or action

    Compulsions are defined by:
    1. Repetitive behaviors or mental acts that the individual feels driven to perform in response to an obsession
    2. The behaviors or mental acts are aimed at preventing or reducing anxiety or distress

    B. The obsessions or compulsions are time-consuming or cause clinically significant distress or impairment in social, occupational, or other important areas of functioning.

    C. The obsessive-compulsive symptoms are not attributable to the physiological effects of a substance or another medical condition.

    D. The disturbance is not better explained by the symptoms of another mental disorder.

    Recurrent Depressive Disorder Classification

    According to ICD-10, recurrent depressive disorder is classified as:
    - F33.0 Recurrent depressive disorder, current episode mild
    - F33.1 Recurrent depressive disorder, current episode moderate
    - F33.2 Recurrent depressive disorder, current episode severe without psychotic symptoms
    - F33.3 Recurrent depressive disorder, current episode severe with psychotic symptoms
    - F33.4 Recurrent depressive disorder, currently in remission
    - F33.8 Other recurrent depressive disorders
    - F33.9 Recurrent depressive disorder, unspecified

    For "Recurrent depressive disorder, currently in remission", the correct coded classification is F33.4.

    Treatment Guidelines for Depression

    The treatment of recurrent depressive disorder involves:
    1. Psychotherapy (CBT, IPT)
    2. Pharmacotherapy (SSRIs, SNRIs)
    3. Combination therapy
    4. Maintenance therapy to prevent relapse

    Remission is defined as a period of at least 2 months with no more than minimal symptoms.
    """

    # Write sample data
    with open(os.path.join(data_dir, "clinical_guidelines.txt"), "w") as f:
        f.write(sample_content)

    print("Sample data created in data/ directory")


def main():
    """Main function to run the RAG QA system."""
    print("=" * 60)
    print("Mini LLM-Powered Question-Answering System (RAG)")
    print("=" * 60)

    # Create sample data if data directory doesn't exist
    if not os.path.exists("data") or not os.listdir("data"):
        print("No data found. Creating sample clinical data...")
        create_sample_data()

    # Initialize query engine
    query_engine = QueryEngine(
        embedding_model="all-MiniLM-L6-v2",
        llm_model="distilgpt2",  # Using smaller model for faster inference
        chunk_size=400,
        overlap=50
    )

    # Setup from documents
    print("\nSetting up query engine...")
    success = query_engine.setup_from_documents("data")

    if not success:
        print("Failed to setup query engine. Please check your documents.")
        return

    # Show system info
    print("\nSystem Information:")
    system_info = query_engine.get_system_info()
    print(json.dumps(system_info, indent=2))

    # Test queries from assignment
    test_queries = [
        "Give me the correct coded classification for the following diagnosis: Recurrent depressive disorder, currently in remission",
        "What are the diagnostic criteria for Obsessive-Compulsive Disorder (OCD)?"
    ]

    print("\n" + "=" * 60)
    print("TESTING WITH ASSIGNMENT QUERIES")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 50)

        result = query_engine.query(query, top_k=3, verbose=True)

        print(f"Answer: {result['answer']}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")

        if 'retrieved_chunks' in result:
            print(f"Retrieved {len(result['retrieved_chunks'])} relevant chunks")
            for j, chunk in enumerate(result['retrieved_chunks'][:2]):  # Show top 2
                print(f"  Chunk {j + 1} (score: {chunk['similarity_score']:.3f}): {chunk['content'][:100]}...")

        print("\n" + "=" * 60)

    # Interactive mode
    print("\nEntering interactive mode. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            user_query = input("\nEnter your question: ").strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not user_query:
                print("Please enter a question.")
                continue

            print("\nProcessing your question...")
            result = query_engine.query(user_query, top_k=5, verbose=False)

            print(f"\nAnswer: {result['answer']}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()