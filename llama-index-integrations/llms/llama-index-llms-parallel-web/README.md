# LlamaIndex LLMs Integration: Parallel

This package provides an integration for using Parallel's Chat API with LlamaIndex.

### Installation

```bash
pip install llama-index-llms-parallel-web
```

### Setup

To get started, you need your Parallel API key. You can either set it as an environment variable `PARALLEL_API_KEY` or pass it directly when initializing the `Parallel` class.

```python
import os
from llama_index.llms.parallel_web import Parallel
from llama_index.core.llms import ChatMessage

# To use your key from an environment variable
# os.environ["PARALLEL_API_KEY"] = "<your-api-key>"

llm = Parallel(
    # Or pass it directly
    # api_key="<your-api-key>",
    model="parallel/speed",
    max_tokens=512,
)
```

### Usage

#### Generate Chat Responses

You can generate a response by passing a list of `ChatMessage` objects.

```python
messages = [
    ChatMessage(
        role="system",
        content="You are a helpful assistant that provides concise answers.",
    ),
    ChatMessage(
        role="user",
        content="What is the significance of the Von Neumann architecture?",
    ),
]
response = llm.chat(messages)
print(response)
```

#### Stream Responses

To stream the response as it's being generated, use the `stream_chat` method.

```python
messages = [
    ChatMessage(
        role="user",
        content="Write a short story about an AI discovering the internet for the first time.",
    ),
]
response_stream = llm.stream_chat(messages)
for chunk in response_stream:
    print(chunk.delta, end="")
```

#### Standard Completion

You can also use the `complete` method for single-prompt completions.

```python
prompt = "Explain the difference between parallel and concurrent programming."
response = llm.complete(prompt)
print(response)
```
