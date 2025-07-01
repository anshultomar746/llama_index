# LlamaIndex Tools Integration: Parallel Web Search

This package provides integration with the [Parallel web search API](https://docs.parallel.ai/resources/search-api) for LlamaIndex tools.

## Installation

```bash
pip install llama-index-tools-parallel-web
```

## Usage

```python
from llama_index.tools.parallel_web import ParallelWebSearchToolSpec

# Initialize the tool with your Parallel API key
tool_spec = ParallelWebSearchToolSpec(api_key="your_parallel_api_key")

# Use objective-based search
results = tool_spec.parallel_search(
    objective="When was the United Nations established? Prefer UN's websites.",
    max_results=5,
)

# Use query-based search
results = tool_spec.parallel_search(
    search_queries=["Founding year UN", "Year of founding United Nations"],
    processor="base",
    max_results=10,
    max_chars_per_result=1500,
)

# Use both objective and queries
results = tool_spec.parallel_search(
    objective="Latest news about space exploration",
    search_queries=["SpaceX latest mission", "NASA recent discoveries"],
    processor="pro",
    max_results=8,
)
```

## Features

- **Speed**: Single API call replaces traditional search → scrape → extract pipeline
- **Token Efficiency**: Returns compressed, structured text optimized for LLMs
- **Flexible**: Configure result count and excerpt length
- **Quality**: Built on web-scale index with advanced ranking

## API Parameters

- `objective`: Natural-language description of research goal (optional)
- `search_queries`: List of search queries to guide the search (optional, max 5)
- `processor`: Either "base" (faster, cheaper) or "pro" (higher quality)
- `max_results`: Maximum number of results (up to 40)
- `max_chars_per_result`: Maximum characters per result (minimum 100)

At least one of `objective` or `search_queries` must be provided.

## Requirements

- Python >=3.9
- A valid Parallel API key

## License

MIT
