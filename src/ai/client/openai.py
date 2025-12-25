from collections.abc import Mapping

import httpx
from openai import AsyncOpenAI, OpenAI

from ai.constants import MAX_API_RETRIES


class OpenAIClient(OpenAI):
    def __init__(
        self,
        api_key: str = "sk-no-key-required",
        base_url: str | httpx.URL | None = None,
        timeout: float | int | None = None,
        max_retries: int = MAX_API_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.Client | None = None,
    ):
        if base_url is None:
            base_url = "http://localhost:5000/v1"

        if not base_url.rstrip("/").endswith("/v1"):
            base_url = str(base_url).rstrip("/") + "/v1"

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
        )


class AsyncOpenAIClient(AsyncOpenAI):
    def __init__(
        self,
        api_key: str = "sk-no-key-required",
        base_url: str | httpx.URL | None = None,
        websocket_base_url: str | httpx.URL | None = None,
        timeout: float | int | None = None,
        max_retries: int = MAX_API_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        if base_url is None:
            base_url = "http://localhost:5000/v1"

        if not base_url.rstrip("/").endswith("/v1"):
            base_url = str(base_url).rstrip("/") + "/v1"

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            websocket_base_url=websocket_base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
        )
