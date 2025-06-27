import json
from collections.abc import Callable, Sequence
from typing import Any, Optional
from os import getenv

import aiohttp
import httpx
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.llm import LLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.openai.utils import to_openai_message_dicts


class ParallelWeb(LLM):
    """
    Parallel Web LLM.

    The Parallel Chat API is a low latency web research API that returns
    OpenAI ChatCompletions compatible streaming text and JSON. The Chat API
    is designed for interactive workflows where speed is paramount.

    Examples:
        `pip install llama-index-llms-parallel-web`

        ```python
        from llama_index.llms.parallel_web import ParallelWeb
        from llama_index.core.llms import ChatMessage

        parallel_api_key = "your-parallel-api-key"

        llm = ParallelWeb(
            api_key=parallel_api_key, model="speed", temperature=0.5
        )

        messages_dict = [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": "What does Parallel Web Systems do?"},
        ]
        messages = [ChatMessage(**msg) for msg in messages_dict]

        response = llm.chat(messages)
        print(str(response))
        ```

    """

    model: str = Field(
        default="speed",
        description="The Parallel Web model to use. Currently only 'speed' is supported.",
    )
    temperature: float = Field(
        default=0.2,
        description="The temperature to use during generation. Note: This parameter is ignored by the API but kept for compatibility.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate. Note: This parameter is ignored by the API but kept for compatibility.",
    )
    context_window: Optional[int] = Field(
        default=None,
        description="The context window to use during generation.",
    )
    api_key: Optional[str] = Field(
        description="The Parallel Web API key.",
        exclude=True,
    )
    api_base: str = Field(
        default="https://beta.parallel.ai",
        description="The base URL for Parallel Web API.",
    )
    additional_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Parallel Web API."
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries."
    )
    headers: dict[str, str] = Field(
        default_factory=dict, description="Headers for API requests."
    )
    is_chat_model: bool = Field(
        default=True,
        description="Whether this is a chat model or not. Default is True.",
    )
    timeout: float = Field(default=30.0, description="HTTP Timeout")

    def __init__(
        self,
        model: str = "speed",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = "https://beta.parallel.ai",
        additional_kwargs: Optional[dict[str, Any]] = None,
        max_retries: int = 10,
        context_window: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or getenv("PARALLEL_API_KEY")
        additional_kwargs = additional_kwargs or {}
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
        }
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            callback_manager=callback_manager,
            api_key=api_key,
            api_base=api_base,
            headers=headers,
            context_window=context_window,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            timeout=timeout,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "parallel_web_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=(
                self.context_window
                if self.context_window is not None
                else self._get_context_window()
            ),
            num_output=self.max_tokens or -1,
            is_chat_model=self.is_chat_model,
            model_name=self.model,
        )

    def _get_context_window(self) -> int:
        """
        The Parallel Chat API uses the 'speed' model which has a large context window.
        Since the exact context window isn't specified in the documentation,
        we use a reasonable default based on modern LLMs.
        """
        # Default context window for the speed model
        return 128000

    def _get_all_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Get all data for the request as a dictionary."""
        base_kwargs = {
            "model": self.model,
        }
        # Only include supported parameters
        # Temperature, max_tokens, etc. are ignored by the API according to docs
        return {**base_kwargs, **self.additional_kwargs, **kwargs}

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        url = f"{self.api_base}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        payload = {
            "messages": messages,
            **self._get_all_kwargs(**kwargs),
        }
        response = requests.post(
            url, json=payload, headers=self.headers, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        return CompletionResponse(
            text=data["choices"][0]["message"]["content"], raw=data
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1))
        def _complete_retry():
            return self._complete(prompt, **kwargs)

        return _complete_retry()

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        url = f"{self.api_base}/chat/completions"
        message_dicts = to_openai_message_dicts(messages)
        payload = {
            "messages": message_dicts,
            **self._get_all_kwargs(**kwargs),
        }
        response = requests.post(
            url, json=payload, headers=self.headers, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        message = ChatMessage(
            role="assistant", content=data["choices"][0]["message"]["content"]
        )
        return ChatResponse(message=message, raw=data)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1))
        def _chat_retry():
            return self._chat(messages, **kwargs)

        return _chat_retry()

    async def _acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        url = f"{self.api_base}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        payload = {
            "messages": messages,
            **self._get_all_kwargs(**kwargs),
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=self.headers, timeout=self.timeout
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return CompletionResponse(
                    text=data["choices"][0]["message"]["content"], raw=data
                )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1))
        async def _acomplete_retry(prompt, **kwargs):
            return await self._acomplete(prompt, **kwargs)

        return await _acomplete_retry(prompt, **kwargs)

    async def _achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        url = f"{self.api_base}/chat/completions"
        message_dicts = to_openai_message_dicts(messages)
        payload = {
            "messages": message_dicts,
            **self._get_all_kwargs(**kwargs),
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=self.headers, timeout=self.timeout
            ) as response:
                response.raise_for_status()
                data = await response.json()
                message = ChatMessage(
                    role="assistant", content=data["choices"][0]["message"]["content"]
                )
                return ChatResponse(message=message, raw=data)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1))
        async def _achat_retry():
            return await self._achat(messages, **kwargs)

        return await _achat_retry()

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        url = f"{self.api_base}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1))
        def make_request():
            return requests.post(
                url,
                json={
                    "messages": messages,
                    "stream": True,
                    **self._get_all_kwargs(**kwargs),
                },
                headers=self.headers,
                timeout=self.timeout,
                stream=True,
            )

        def gen() -> CompletionResponseGen:
            response = make_request()
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]
                        if line.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(line)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield CompletionResponse(delta=delta, raw=data)
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        return self._stream_complete(prompt, **kwargs)

    async def _astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        url = f"{self.api_base}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1))
        async def make_request():
            async with httpx.AsyncClient() as client:
                return await client.post(
                    url,
                    json={
                        "messages": messages,
                        "stream": True,
                        **self._get_all_kwargs(**kwargs),
                    },
                    headers=self.headers,
                    timeout=self.timeout,
                )

        async def gen() -> CompletionResponseAsyncGen:
            response = await make_request()
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    if line.startswith("data: "):
                        line = line[6:]
                        if line.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(line)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield CompletionResponse(delta=delta, raw=data)
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        return await self._astream_complete(prompt, **kwargs)

    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        url = f"{self.api_base}/chat/completions"
        message_dicts = to_openai_message_dicts(messages)

        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1))
        def make_request():
            return requests.post(
                url,
                json={
                    "messages": message_dicts,
                    "stream": True,
                    **self._get_all_kwargs(**kwargs),
                },
                headers=self.headers,
                timeout=self.timeout,
                stream=True,
            )

        def gen() -> ChatResponseGen:
            response = make_request()
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]
                        if line.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(line)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield ChatResponse(
                                    message=ChatMessage(role="assistant", content=""),
                                    delta=delta,
                                    raw=data,
                                )
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        return gen()

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        return self._stream_chat(messages, **kwargs)

    async def _astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        url = f"{self.api_base}/chat/completions"
        message_dicts = to_openai_message_dicts(messages)

        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1))
        async def make_request():
            async with httpx.AsyncClient() as client:
                return await client.post(
                    url,
                    json={
                        "messages": message_dicts,
                        "stream": True,
                        **self._get_all_kwargs(**kwargs),
                    },
                    headers=self.headers,
                    timeout=self.timeout,
                )

        async def gen():
            response = await make_request()
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    if line.startswith("data: "):
                        line = line[6:]
                        if line.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(line)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield ChatResponse(
                                    message=ChatMessage(role="assistant", content=""),
                                    delta=delta,
                                    raw=data,
                                )
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        return gen()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        return await self._astream_chat(messages, **kwargs)
