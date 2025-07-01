from typing import Any, Dict, Optional

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.bridge.pydantic import Field
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.llms.openai_like import OpenAILike

# Default API endpoint for Parallel Chat API (Beta)
DEFAULT_API_BASE = "https://beta.parallel.ai"
# Default model to use (as per Chat API documentation)
DEFAULT_MODEL = "speed"


class ParallelWeb(OpenAILike):
    """
    Parallel Chat API LLM integration.

    The Parallel Chat API is a low latency web research API that returns OpenAI
    ChatCompletions compatible streaming text and JSON responses. It brings
    real-time web research to interactive AI applications.

    To use this integration, you need to have a Parallel API key.
    You can set the API key as an environment variable `PARALLEL_API_KEY`
    or pass it directly to the constructor.

    Features:
        - OpenAI-compatible Chat API with web research capabilities
        - Low latency responses (3 second p50 TTFT with streaming)
        - JSON schema support for structured outputs
        - Real-time web data integration

    Examples:
        `pip install llama-index-llms-parallel-web`

        ```python
        from llama_index.llms.parallel_web import ParallelWeb

        # Basic usage
        llm = ParallelWeb(
            api_key="<your-api-key>",
            model="speed",
            max_tokens=512,
        )

        response = llm.complete("What does Parallel Web Systems do?")
        print(str(response))

        # Chat with web research
        from llama_index.core.llms import ChatMessage

        messages = [
            ChatMessage(role="user", content="What are the latest developments in AI?")
        ]
        response = llm.chat(messages)
        print(response)
        ```

    Note:
        The Parallel Chat API is currently in beta with a rate limit of 30 requests
        per minute. Contact Parallel for production capacity.

    """

    model: str = Field(default=DEFAULT_MODEL, description="The Parallel model to use.")
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    is_chat_model: bool = Field(
        default=True,
        description=LLMMetadata.model_fields["is_chat_model"].description,
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        api_key: Optional[str] = None,
        api_base: str = DEFAULT_API_BASE,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Parallel LLM client.

        Args:
            model (str): The model to use.
            temperature (float): The temperature for sampling.
            max_tokens (int): The maximum number of tokens to generate.
            api_key (Optional[str]): The API key.
            api_base (str): The base URL for the API.
            additional_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments.
            max_retries (int): The maximum number of retries for API requests.

        """
        additional_kwargs = additional_kwargs or {}

        # Retrieve API key from params or environment
        api_key = get_from_param_or_env("api_key", api_key, "PARALLEL_API_KEY")

        # Use model as-is for the new Chat API (no prefix needed)
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get the class name."""
        return "ParallelWeb"
