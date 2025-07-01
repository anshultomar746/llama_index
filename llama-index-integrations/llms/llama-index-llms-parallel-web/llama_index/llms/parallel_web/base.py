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

# Default API endpoint for Parallel
DEFAULT_API_BASE = "https://api.parallel.ai/v1"
# Default model to use
DEFAULT_MODEL = "parallel/speed"


class ParallelWeb(OpenAILike):
    """
    Parallel LLM.

    To use this integration, you need to have a Parallel API key.
    You can set the API key as an environment variable `PARALLEL_API_KEY`
    or pass it directly to the constructor.

    Examples:
        `pip install llama-index-llms-parallel`

        ```python
        from llama_index.llms.parallel_web import ParallelWeb

        llm = ParallelWeb(
            api_key="<your-api-key>",
            model="parallel/speed",
            max_tokens=512,
        )

        response = llm.complete("What is the history of parallel computing?")
        print(str(response))
        ```

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
        parallel_model_name = model[len("parallel/") :]
        super().__init__(
            model=parallel_model_name,
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
