import logging
from collections.abc import Generator

import httpx

from .models import (
    SwarmGeneratedImage,
    SwarmGenerationResponse,
    SwarmSessionInfo,
    SwarmT2IParams,
    SwarmUIError,
    SwarmUISettings,
    SwarmUIStatus,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SwarmUIAuth(httpx.Auth):
    """HTTPX Auth class for SwarmUI API using API token."""

    # GetNewSession returns session info in the body
    requires_response_body = True

    def __init__(self, api_token: str | None):
        self.api_token = api_token
        self.session_info: SwarmSessionInfo | None = None

    @property
    def session_id(self) -> str | None:
        return self.session_info.session_id if self.session_info else None

    @session_id.setter
    def session_id(self, value: str | None):
        if self.session_info:
            self.session_info.session_id = value

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        if self.api_token:
            request.headers["Authorization"] = f"Bearer {self.api_token}"
        if not self.session_id:
            refresh_response = yield self.build_refresh_request(request)
            self.update_session_info(refresh_response)

        request.headers["X-Session-Id"] = self.session_id
        response = yield request

        if response.status_code == 401 and response.json().get("error_id", None) == "invalid_session_id":
            logger.info("Session ID invalid, refreshing session...")
            refresh_response = yield self.build_refresh_request(request)
            self.update_session_info(refresh_response)
            request.headers["X-Session-Id"] = self.session_id
            yield request

    def build_refresh_request(self, request: httpx.Request) -> httpx.Request:
        url = request.url.copy_with(path="/API/GetNewSession")
        headers = request.headers.copy()
        headers.pop("X-Session-Id", None)
        headers.pop("Content-Length", None)
        return httpx.Request("POST", url, json={}, headers=headers)

    def update_session_info(self, response: httpx.Response):
        data = SwarmSessionInfo.model_validate_json(response.text)
        if data.error_id:
            raise SwarmUIError(message=data.error, error_id=data.error_id, status_code=response.status_code)
        self.session_info = data


class SwarmUIClient:
    def __init__(
        self,
        settings: SwarmUISettings | None = None,
        **client_kwargs,
    ):
        if settings is None:
            # custom pydantic settings class value autoloading confuses pylance, so ignore type here
            settings = SwarmUISettings()  # type: ignore

        self.settings = settings
        self.auth = SwarmUIAuth(self.settings.api_token)
        self.cookies = httpx.Cookies()

        if self.settings.verify_ssl:
            self.verify = self.settings.verify_ssl
        else:
            self.verify = True if self.settings.base_url.startswith("https") else False

        self.client = httpx.Client(
            base_url=self.settings.base_url,
            auth=self.auth,
            headers=self.headers,
            cookies=self.cookies,
            verify=self.verify,
            timeout=self.settings.default_timeout,
            follow_redirects=True,
            **client_kwargs,
        )

    def close(self):
        """Close the underlying HTTP clients."""
        if self.client and not self.client.is_closed:
            self.client.close()

    @property
    def headers(self) -> dict[str, str]:
        """Get the default headers for the client."""
        return {"User-Agent": self.settings.user_agent, "Accept": "application/json"}

    @property
    def session_info(self) -> SwarmSessionInfo | None:
        """Get the current session info."""
        return self.auth.session_info

    def get_current_status(self) -> SwarmUIStatus:
        """Get the current status of the SwarmUI server."""
        response = self._send_request("GetCurrentStatus")
        try:
            response.raise_for_status()
            return SwarmUIStatus.model_validate_json(response.content)
        except Exception as e:
            logger.exception(f"Failed to parse GetCurrentStatus response: {response.content.decode()}")
            raise e

    def generate_images(
        self,
        params: SwarmT2IParams | None = None,
        prompt: str | None = None,
        images: int = 1,
        wrap_prompt: bool = False,
    ) -> SwarmGeneratedImage | list[SwarmGeneratedImage]:
        """Generate an image using the SwarmUI API with the given parameters."""
        if not prompt and not params:
            raise ValueError("Either prompt or params must be provided.")
        # load default params if none provided
        params = params if params else self.settings.default_params.model_copy()

        # set prompt if provided
        if prompt:
            params.prompt = prompt
        # wrap prompt if needed
        if wrap_prompt:
            params.prompt = self.settings.wrap_prompt(params.prompt)

        payload = params.model_dump(exclude_none=True, mode="json")
        payload["images"] = images

        response = self._send_request("GenerateText2Image", payload)
        if not response.content:
            response.raise_for_status()

        response = SwarmGenerationResponse.model_validate_json(response.content)
        response.raise_for_error()
        n_generated = len(response.images)
        logger.info(f"Generation request successful, retrieving {n_generated} images...")

        # ok now retrieve image(s)
        generated_images = []
        for idx, image_path in enumerate(response.images):
            try:
                logger.debug(f"Retrieving image {idx + 1}/{n_generated}: {image_path}")
                image_bytes = self._get_file_content(image_path)
                generated_image = SwarmGeneratedImage(
                    server_path=image_path, image_bytes=image_bytes, params=params
                )
                generated_image.save_file()  # save to disk
                generated_images.append(generated_image)
            except Exception:
                logger.exception(f"Error retrieving image from {image_path}")
        logger.debug(f"Retrieved {len(generated_images)} images")

        return generated_images

    def _send_request(
        self,
        endpoint: str,
        payload: dict | None = None,
    ) -> httpx.Response:
        """Send an HTTP POST request to the SwarmUI API."""
        if not payload:
            payload = {}
        request = self.client.build_request("POST", f"/API/{endpoint}", json=payload)
        logger.debug(f"Sending request to {endpoint} with payload: {payload}")
        return self.client.send(request)

    def _get_file_content(
        self,
        path: str,
    ) -> bytes:
        """Send a GET request to the SwarmUI server. Used for retrieving images."""
        request = self.client.build_request("GET", path)
        logger.debug(f"Sending request for file content at {path}")
        response = self.client.send(request)
        response.raise_for_status()
        return response.content


class SwarmUIClientAsync(SwarmUIClient):
    def __init__(
        self,
        settings: SwarmUISettings | None = None,
        **client_kwargs,
    ):
        super().__init__(settings=settings, **client_kwargs)

        self.aclient = httpx.AsyncClient(
            base_url=self.settings.base_url,
            auth=self.auth,
            headers=self.headers,
            cookies=self.cookies,
            verify=self.verify,
            timeout=self.settings.default_timeout,
            follow_redirects=True,
            **client_kwargs,
        )

    async def aclose(self):
        """Close the underlying async HTTP client."""
        if self.aclient and not self.aclient.is_closed:
            await self.aclient.aclose()

    async def get_current_status_async(self) -> SwarmUIStatus:
        """Get the current status of the SwarmUI server."""
        response = await self._send_request_async("GetCurrentStatus")
        try:
            response.raise_for_status()
            payload = await response.aread()
            result = SwarmUIStatus.model_validate_json(payload)
            result.raise_for_error()
            return result
        except Exception:
            logger.exception(f"Failed to parse GetCurrentStatus response: {response.text}")

    async def _send_request_async(
        self,
        endpoint: str,
        payload: dict | None = None,
    ) -> httpx.Response:
        """Send an asynchronous HTTP POST request to the SwarmUI API."""
        if payload is None:
            payload = {}
        request = self.aclient.build_request("POST", f"/API/{endpoint}", json=payload)
        logger.debug(f"Sending async request to {endpoint} with payload: {payload}")
        return await self.aclient.send(request)

    async def _get_file_content_async(
        self,
        path: str,
    ) -> bytes:
        """Send an asynchronous GET request to the SwarmUI server. Used for retrieving images."""
        request = self.aclient.build_request("GET", path)
        logger.debug(f"Sending async request for file content at {path}")
        response = await self.aclient.send(request)
        response.raise_for_status()
        return await response.aread()
