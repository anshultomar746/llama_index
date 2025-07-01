"""Basic usage example for Parallel web search tool."""

import os
from llama_index.tools.parallel_web import ParallelWebSearchToolSpec

def main():
    # Get API key from environment variable
    api_key = os.getenv("PARALLEL_API_KEY")
    if not api_key:
        print("Please set PARALLEL_API_KEY environment variable")
        return

    # Initialize the tool
    tool_spec = ParallelWebSearchToolSpec(api_key=api_key)

    # Example 1: Objective-based search
    print("Example 1: Objective-based search")
    print("=" * 50)
    results = tool_spec.parallel_search(
        objective="When was the United Nations established? Prefer UN's websites.",
        max_results=3,
        processor="base"
    )

    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Title: {doc.extra_info.get('title', 'N/A')}")
        print(f"URL: {doc.extra_info.get('url', 'N/A')}")
        print(f"Content: {doc.text[:200]}...")

    # Example 2: Query-based search
    print("\n\nExample 2: Query-based search")
    print("=" * 50)
    results = tool_spec.parallel_search(
        search_queries=["SpaceX latest mission", "NASA recent discoveries"],
        max_results=5,
        max_chars_per_result=1000,
        processor="base"
    )

    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Title: {doc.extra_info.get('title', 'N/A')}")
        print(f"URL: {doc.extra_info.get('url', 'N/A')}")
        print(f"Content: {doc.text[:150]}...")

    # Example 3: Combined objective and queries
    print("\n\nExample 3: Combined objective and queries")
    print("=" * 50)
    results = tool_spec.parallel_search(
        objective="Latest developments in artificial intelligence and machine learning",
        search_queries=["AI breakthroughs 2024", "machine learning advances"],
        max_results=4,
        processor="base"
    )

    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Title: {doc.extra_info.get('title', 'N/A')}")
        print(f"URL: {doc.extra_info.get('url', 'N/A')}")
        print(f"Content: {doc.text[:150]}...")

if __name__ == "__main__":
    main()
