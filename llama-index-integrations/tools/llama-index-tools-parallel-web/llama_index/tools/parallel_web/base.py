"""Parallel web search tool spec."""

import requests
from typing import Dict, List, Optional
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

PARALLEL_SEARCH_URL_ENDPOINT = "https://api.parallel.ai/alpha/search"


class ParallelWebSearchToolSpec(BaseToolSpec):
    """
    Parallel web search tool spec.

    Provides access to Parallel's web search API which streamlines the traditional
    search → scrape → extract pipeline into a single, low-latency API—reducing
    token overhead and integration effort.
    """

    spec_functions = ["parallel_search"]

    def __init__(self, api_key: str) -> None:
        """
        Initialize with API key.

        Args:
            api_key (str): The Parallel API key for authentication.

        """
        self.api_key = api_key

    def _make_request(self, data: Dict) -> Dict:
        """
        Make a request to the Parallel Search API.

        Args:
            data (dict): The request payload to be sent to the API.

        Returns:
            Dict: The JSON response from the API.

        """
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }
        response = requests.post(
            PARALLEL_SEARCH_URL_ENDPOINT, json=data, headers=headers
        )
        response.raise_for_status()
        return response.json()

    def parallel_search(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
        processor: str = "base",
        max_results: int = 10,
        max_chars_per_result: int = 1500,
    ) -> List[Document]:
        """
        Make a query to the Parallel Search API to receive search results.

        Args:
            objective (Optional[str]): Natural-language description of what the web research goal is.
                                     Include any source or freshness guidance.
            search_queries (Optional[List[str]]): Optional search queries to guide the search.
                                                Maximum 5 queries, each up to 200 characters.
            processor (str): Either "base" (faster, lower cost) or "pro" (higher quality, freshness).
                           Defaults to "base".
            max_results (int): Maximum number of search results (up to 40). Defaults to 10.
            max_chars_per_result (int): Maximum characters per search result (minimum 100).
                                      Defaults to 1500.

        Returns:
            List[Document]: A list of documents containing search results with excerpts.

        """
        if objective is None and search_queries is None:
            raise ValueError(
                "At least one of 'objective' or 'search_queries' must be provided"
            )

        search_params = {
            "processor": processor,
            "max_results": max_results,
            "max_chars_per_result": max_chars_per_result,
        }

        if objective:
            search_params["objective"] = objective

        if search_queries:
            if len(search_queries) > 5:
                raise ValueError("Maximum 5 search queries allowed")
            for query in search_queries:
                if len(query) > 200:
                    raise ValueError("Each search query must be 200 characters or less")
            search_params["search_queries"] = search_queries

        response = self._make_request(search_params)

        documents = []
        for result in response.get("results", []):
            # Join excerpts with newlines to create the document text
            text_content = "\n".join(result.get("excerpts", []))

            documents.append(
                Document(
                    text=text_content,
                    extra_info={
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "search_id": response.get("search_id", ""),
                    },
                )
            )

        return documents
