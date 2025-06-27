# LlamaIndex Llms Integration: Parallel Web

The Parallel Web integration for LlamaIndex allows you to tap into real-time web research powered by the Parallel Chat API. This integration supports synchronous and asynchronous chat completions—as well as streaming responses.

## Installation

To install the required packages, run:

```bash
%pip install llama-index-llms-parallel-web
!pip install llama-index
```

## Setup

### Import Libraries and Configure API Key

Please refer to the official Parallel [API documentation](https://docs.parallel.ai/resources/chat-api) to get started. You can follow the steps outlined to generate your API key.

Import the necessary libraries and set your Parallel API key:

```python
from llama_index.llms.parallel_web import ParallelWeb

parallel_api_key = "your-parallel-api-key"  # Replace with your actual API key
```

### Initialize the Parallel Web LLM

Create an instance of the Parallel Web LLM with your API key and desired model settings:

```python
llm = ParallelWeb(api_key=parallel_api_key, model="speed")
```

## Chat Example

### Sending a Chat Message

You can send a chat message using the `chat` method. Here's how to do that:

```python
from llama_index.core.llms import ChatMessage

messages_dict = [
    {"role": "system", "content": "Be precise and concise."},
    {
        "role": "user",
        "content": "What does Parallel Web Systems do?",
    },
]

messages = [ChatMessage(**msg) for msg in messages_dict]

# Obtain a response from the model
response = llm.chat(messages)
print(response)
```

### Async Chat

For asynchronous conversation processing, use the `achat` method to send messages and await the response:

```python
response = await llm.achat(messages)
print(response)
```

### Stream Chat

For cases where you want to receive a response token by token in real time, use the `stream_chat` method:

```python
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
```

### Async Stream Chat

Similarly, for asynchronous streaming, the `astream_chat` method provides a way to process response deltas asynchronously:

```python
resp = await llm.astream_chat(messages)
async for delta in resp:
    print(delta.delta, end="")
```

### JSON Response Format

The Parallel Chat API supports structured JSON responses. You can specify a JSON schema for the response:

```python
response = llm.chat(
    messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "reasoning_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Think step by step to arrive at the answer",
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
    },
)
print(response)
```

### Tool calling

Parallel Web models can easily be wrapped into a llamaindex tool so that it can be called as part of your data processing or conversational workflows. This tool uses real-time web research powered by Parallel Web.

Below is an example of how to define and register the tool:

```python
from llama_index.core.tools import FunctionTool
from llama_index.llms.parallel_web import ParallelWeb
from llama_index.core.llms import ChatMessage


def query_parallel_web(query: str) -> str:
    """
    Queries the Parallel Web API via the LlamaIndex integration.

    This function instantiates a Parallel Web LLM with default settings
    (using model "speed" for low latency), wraps the query into a ChatMessage,
    and returns the generated response content with real-time web research.
    """
    parallel_api_key = (
        "your-parallel-api-key"  # Replace with your actual API key
    )

    llm = ParallelWeb(
        api_key=parallel_api_key,
        model="speed",
    )

    messages = [ChatMessage(role="user", content=query)]
    response = llm.chat(messages)
    return response.message.content


# Create the tool from the query_parallel_web function
query_parallel_web_tool = FunctionTool.from_defaults(fn=query_parallel_web)
```

### Performance and Rate Limits

- **Performance**: With `stream=true`, achieves 3 second p50 TTFT (median time to first token)
- **Default Rate Limit**: 30 requests per minute
- **Use Cases**: Chat interfaces, interactive tools requiring real-time web research

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/parallel_web/
