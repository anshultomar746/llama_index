import inspect
import os
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from tenacity import RetryError

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.llms.parallel_web import ParallelWeb


@pytest.fixture()
def parallel_web_llm():
    api_key = os.getenv("PARALLEL_API_KEY")
    if api_key is None:
        pytest.skip("PARALLEL_API_KEY not set in environment")
    return ParallelWeb(api_key=api_key)


@pytest.fixture()
def mock_parallel_web_llm():
    return ParallelWeb(api_key="test")


def test_get_context_window():
    llm = ParallelWeb(api_key="dummy", model="speed")
    assert llm._get_context_window() == 128000


def test_get_all_kwargs():
    llm = ParallelWeb(
        api_key="dummy", additional_kwargs={"foo": "bar"}, temperature=0.7
    )
    all_kwargs = llm._get_all_kwargs(custom=123)
    assert all_kwargs["foo"] == "bar"
    assert all_kwargs["custom"] == 123
    assert all_kwargs["model"] == "speed"


def test_chat(parallel_web_llm):
    messages = [
        ChatMessage(role="system", content="Be precise and concise."),
        ChatMessage(role="user", content="What does Parallel Web Systems do?"),
    ]
    response = parallel_web_llm.chat(messages)
    assert isinstance(response, ChatResponse)
    assert response.message.content.strip()


def test_complete(parallel_web_llm):
    prompt = "Parallel Web Systems is a company that provides"
    response = parallel_web_llm.complete(prompt)
    assert isinstance(response, CompletionResponse)
    assert response.text.strip()


def test_stream_chat(parallel_web_llm):
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(
            role="user", content="Name the first 5 elements in the periodic table."
        ),
    ]
    stream = parallel_web_llm.stream_chat(messages)
    assert inspect.isgenerator(stream)
    response = ""
    for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        assert chunk.delta is not None
        response += chunk.delta
    assert response.strip()


def test_stream_complete(parallel_web_llm):
    prompt = "List the first 5 planets in the solar system:"
    stream = parallel_web_llm.stream_complete(prompt)
    assert inspect.isgenerator(stream)
    response = ""
    for chunk in stream:
        assert isinstance(chunk, CompletionResponse)
        assert chunk.delta is not None
        response += chunk.delta
    assert response.strip()


@pytest.mark.asyncio
async def test_achat(parallel_web_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="What is the largest planet in our solar system?",
        ),
    ]
    response = await parallel_web_llm.achat(messages)
    assert isinstance(response, ChatResponse)
    assert response.message.content.strip()


@pytest.mark.asyncio
async def test_acomplete(parallel_web_llm):
    prompt = "The largest planet in our solar system is"
    response = await parallel_web_llm.acomplete(prompt)
    assert isinstance(response, CompletionResponse)
    assert response.text.strip()


@pytest.mark.asyncio
async def test_astream_chat(parallel_web_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="Name the first 5 elements in the periodic table.",
        ),
    ]
    stream = await parallel_web_llm.astream_chat(messages)
    assert isinstance(stream, AsyncIterator)
    response = ""
    async for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        assert chunk.delta is not None
        response += chunk.delta
    assert response.strip()


@pytest.mark.asyncio
async def test_astream_complete(parallel_web_llm):
    prompt = "List the first 5 elements in the periodic table:"
    stream = await parallel_web_llm.astream_complete(prompt)
    assert isinstance(stream, AsyncIterator)
    response = ""
    async for chunk in stream:
        assert isinstance(chunk, CompletionResponse)
        assert chunk.delta is not None
        response += chunk.delta
    assert response.strip()


def test_chat_mock(mock_parallel_web_llm):
    with patch.object(
        mock_parallel_web_llm,
        "_chat",
        return_value=ChatResponse(
            message=ChatMessage(role="assistant", content="mock")
        ),
    ) as mock_chat:
        messages = [ChatMessage(role="user", content="Hi")]
        result = mock_parallel_web_llm.chat(messages)
        assert result.message.content == "mock"
        mock_chat.assert_called_once_with(messages)


def test_complete_mock(mock_parallel_web_llm):
    with patch.object(
        mock_parallel_web_llm, "_complete", return_value=CompletionResponse(text="mock")
    ) as mock_complete:
        result = mock_parallel_web_llm.complete("hello")
        assert result.text == "mock"
        mock_complete.assert_called_once_with("hello")


@pytest.mark.asyncio
async def test_achat_mock(mock_parallel_web_llm):
    with patch.object(
        mock_parallel_web_llm,
        "_achat",
        new=AsyncMock(
            return_value=ChatResponse(
                message=ChatMessage(role="assistant", content="mock")
            )
        ),
    ) as mock_achat:
        messages = [ChatMessage(role="user", content="Hi")]
        result = await mock_parallel_web_llm.achat(messages)
        assert result.message.content == "mock"
        mock_achat.assert_called_once_with(messages)


@pytest.mark.asyncio
async def test_acomplete_mock(mock_parallel_web_llm):
    with patch.object(
        mock_parallel_web_llm,
        "_acomplete",
        new=AsyncMock(return_value=CompletionResponse(text="mock")),
    ) as mock_acomplete:
        result = await mock_parallel_web_llm.acomplete("hello")
        assert result.text == "mock"
        mock_acomplete.assert_called_once_with("hello")


def test_stream_chat_mock(mock_parallel_web_llm):
    def mock_generator():
        yield ChatResponse(
            message=ChatMessage(role="assistant", content=""), delta="chunk1"
        )
        yield ChatResponse(
            message=ChatMessage(role="assistant", content=""), delta="chunk2"
        )

    with patch.object(
        mock_parallel_web_llm, "_stream_chat", return_value=mock_generator()
    ) as mock_stream_chat:
        messages = [ChatMessage(role="user", content="Hi")]
        result = list(mock_parallel_web_llm.stream_chat(messages))
        assert len(result) == 2
        assert result[0].delta == "chunk1"
        assert result[1].delta == "chunk2"
        mock_stream_chat.assert_called_once_with(messages)


def test_stream_complete_mock(mock_parallel_web_llm):
    def mock_generator():
        yield CompletionResponse(text="", delta="chunk1")
        yield CompletionResponse(text="", delta="chunk2")

    with patch.object(
        mock_parallel_web_llm, "_stream_complete", return_value=mock_generator()
    ) as mock_stream_complete:
        result = list(mock_parallel_web_llm.stream_complete("hello"))
        assert len(result) == 2
        assert result[0].delta == "chunk1"
        assert result[1].delta == "chunk2"
        mock_stream_complete.assert_called_once_with("hello")


@pytest.mark.asyncio
async def test_astream_chat_mock(mock_parallel_web_llm):
    async def mock_agen():
        yield ChatResponse(
            message=ChatMessage(role="assistant", content=""), delta="chunk1"
        )
        yield ChatResponse(
            message=ChatMessage(role="assistant", content=""), delta="chunk2"
        )

    with patch.object(
        mock_parallel_web_llm, "_astream_chat", return_value=mock_agen()
    ) as mock_astream_chat:
        messages = [ChatMessage(role="user", content="Hi")]
        result = []
        async for chunk in await mock_parallel_web_llm.astream_chat(messages):
            result.append(chunk)
        assert len(result) == 2
        assert result[0].delta == "chunk1"
        assert result[1].delta == "chunk2"
        mock_astream_chat.assert_called_once_with(messages)


@pytest.mark.asyncio
async def test_astream_complete_mock(mock_parallel_web_llm):
    async def mock_agen():
        yield CompletionResponse(text="", delta="chunk1")
        yield CompletionResponse(text="", delta="chunk2")

    with patch.object(
        mock_parallel_web_llm, "_astream_complete", return_value=mock_agen()
    ) as mock_astream_complete:
        result = []
        async for chunk in await mock_parallel_web_llm.astream_complete("hello"):
            result.append(chunk)
        assert len(result) == 2
        assert result[0].delta == "chunk1"
        assert result[1].delta == "chunk2"
        mock_astream_complete.assert_called_once_with("hello")


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("complete", ("test",)),
        ("chat", ([ChatMessage(role=MessageRole.USER, content="Hi")],)),
    ],
)
def test_sync_errors(mock_parallel_web_llm, method, args):
    with patch.object(
        mock_parallel_web_llm,
        f"_{method}",
        side_effect=Exception("API Error"),
    ):
        with pytest.raises(RetryError):
            getattr(mock_parallel_web_llm, method)(*args)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("acomplete", ("test",)),
        ("achat", ([ChatMessage(role=MessageRole.USER, content="Hi")],)),
    ],
)
async def test_async_errors(mock_parallel_web_llm, method, args):
    with patch.object(
        mock_parallel_web_llm,
        f"_{method}",
        new=AsyncMock(side_effect=Exception("API Error")),
    ):
        with pytest.raises(RetryError):
            await getattr(mock_parallel_web_llm, method)(*args)
