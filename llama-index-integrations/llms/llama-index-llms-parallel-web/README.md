# LlamaIndex LLMs Integration: Parallel Web

This package provides an integration for using [Parallel's Chat API](https://docs.parallel.ai/resources/chat-api) with LlamaIndex.

The Parallel Chat API is a low latency web research API that returns OpenAI ChatCompletions compatible streaming text and JSON responses. It brings real-time web research to interactive AI applications with a 3 second p50 time-to-first-token.

## Features

- **Real-time web research**: Get up-to-date information from the web
- **OpenAI compatibility**: Drop-in replacement for OpenAI's API
- **Low latency**: 3 second p50 TTFT with streaming
- **JSON schema support**: Structured outputs with response formatting
- **Interactive applications**: Optimized for chat interfaces and tools

## Installation

```bash
pip install llama-index-llms-parallel-web
```

## Setup

To get started, you need your Parallel API key from [Parallel.ai](https://parallel.ai). You can either set it as an environment variable `PARALLEL_API_KEY` or pass it directly when initializing the `ParallelWeb` class.

```python
import os
from llama_index.llms.parallel_web import ParallelWeb
from llama_index.core.llms import ChatMessage

# Option 1: Use environment variable
os.environ["PARALLEL_API_KEY"] = "<your-api-key>"
llm = ParallelWeb(model="speed")

# Option 2: Pass API key directly
llm = ParallelWeb(
    api_key="<your-api-key>",
    model="speed",
    max_tokens=512,
)
```

## Usage

### Basic Chat with Web Research

The main advantage of Parallel's Chat API is real-time web research. Ask questions that require current information:

```python
messages = [
    ChatMessage(
        role="user",
        content="What are the latest developments in artificial intelligence this week?",
    ),
]
response = llm.chat(messages)
print(response)
```

### JSON Response Format

Use structured JSON responses for applications that need formatted data:

```python
from llama_index.core.llms import ChatMessage

# Define a JSON schema for structured responses
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "research_response",
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Step by step reasoning for the answer",
                },
                "answer": {
                    "type": "string",
                    "description": "The direct answer to the question",
                },
                "citations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sources cited to support the answer",
                },
            },
        },
    },
}

# Create LLM with JSON response format
llm = ParallelWeb(
    api_key="<your-api-key>", model="speed", response_format=response_format
)

messages = [
    ChatMessage(role="user", content="What does Parallel Web Systems do?")
]
response = llm.chat(messages)
print(response)
```

### Streaming Responses

For interactive applications, stream responses as they're generated:

```python
messages = [
    ChatMessage(
        role="user",
        content="What are the current trends in renewable energy technology?",
    ),
]

response_stream = llm.stream_chat(messages)
for chunk in response_stream:
    print(chunk.delta, end="", flush=True)
print()  # New line after streaming
```

### Simple Completions

For straightforward prompts, use the `complete` method:

```python
prompt = "Explain the latest breakthroughs in quantum computing."
response = llm.complete(prompt)
print(response)
```

### Web Research Examples

The Chat API excels at questions requiring current information:

```python
# Current events
response = llm.complete("What are today's top technology news stories?")

# Recent developments
response = llm.complete("What are the latest updates from major AI companies?")

# Real-time data
response = llm.complete("What is the current status of SpaceX missions?")
```

## Rate Limits and Performance

- **Rate Limit**: 30 requests per minute (beta)
- **Performance**: 3 second p50 time-to-first-token with streaming
- **Use Cases**: Chat interfaces, interactive tools, real-time research

For production deployments requiring higher throughput, contact [Parallel](https://parallel.ai) for increased capacity.
