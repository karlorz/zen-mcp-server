"""Gemini-compatible API provider implementation."""

import logging
import os
from typing import Optional

from .base import ProviderType
from .gemini import GeminiModelProvider

logger = logging.getLogger(__name__)


class GeminiCompatibleProvider(GeminiModelProvider):
    """Gemini-compatible API provider for custom Gemini hosts.

    This provider extends the standard GeminiModelProvider to support custom
    Gemini API endpoints while maintaining all the same functionality and model
    configurations from the base Gemini provider.

    Activated when GEMINI_API_HOST environment variable is set.
    """

    FRIENDLY_NAME = "Gemini Compatible API"

    def __init__(self, api_key: str, base_url: str = "", **kwargs):
        """Initialize Gemini-compatible provider for custom Gemini hosts.

        This provider supports custom Gemini API endpoints that are compatible
        with the Google Gemini API interface but hosted on different infrastructure.

        Args:
            api_key: API key for the custom Gemini endpoint. Falls back to
                    GEMINI_API_KEY environment variable if not provided.
            base_url: Base URL for the custom Gemini API endpoint. Falls back to
                     GEMINI_API_HOST environment variable if not provided.
            **kwargs: Additional configuration passed to parent Gemini provider

        Raises:
            ValueError: If no base_url is provided via parameter or environment variable
        """
        # Fall back to environment variables only if not provided
        if not base_url:
            base_url = os.getenv("GEMINI_API_HOST", "")
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY", "")

        if not base_url:
            raise ValueError(
                "Gemini API host URL must be provided via base_url parameter or GEMINI_API_HOST environment variable"
            )

        if not api_key:
            raise ValueError(
                "Gemini API key must be provided via api_key parameter or GEMINI_API_KEY environment variable"
            )

        logging.info(f"Initializing Gemini-compatible provider with custom host: {base_url}")

        # Store custom host for client initialization
        self.custom_host = base_url

        # Initialize parent with API key
        super().__init__(api_key, **kwargs)

    @property
    def client(self):
        """Lazy initialization of Gemini client with custom host."""
        if self._client is None:
            from google import genai

            try:
                # Configure client with custom base URL using http_options
                from google.genai.client import HttpOptions

                http_options = HttpOptions(base_url=self.custom_host)
                self._client = genai.Client(api_key=self.api_key, http_options=http_options)
                logging.info(f"Successfully initialized Gemini client with custom host: {self.custom_host}")

            except Exception as e:
                logging.error(f"Failed to initialize Gemini client with custom host {self.custom_host}: {e}")

                # Fallback: If custom endpoint configuration fails, log a warning and use standard client
                # This allows graceful degradation while still providing access to Gemini models
                logging.warning(
                    f"Custom Gemini host configuration failed, falling back to standard Gemini client. "
                    f"Custom host '{self.custom_host}' may not be supported by the current genai SDK version."
                )
                self._client = genai.Client(api_key=self.api_key)

        return self._client

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        # Return GOOGLE type since this is still a Gemini provider, just with custom host
        return ProviderType.GOOGLE

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if the model name is supported and allowed.

        Uses the same validation logic as the parent Gemini provider since
        this is still providing Gemini models, just from a custom host.
        """
        return super().validate_model_name(model_name)

    def get_capabilities(self, model_name: str):
        """Get capabilities for a Gemini model from custom host.

        Uses the same model capabilities as the parent Gemini provider since
        the models should have the same capabilities regardless of host.
        """
        capabilities = super().get_capabilities(model_name)

        # Update friendly name to indicate custom host
        capabilities.friendly_name = f"{self.FRIENDLY_NAME} ({capabilities.model_name})"

        return capabilities

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode.

        Uses the same logic as parent Gemini provider since model capabilities
        should be consistent regardless of the hosting infrastructure.
        """
        return super().supports_thinking_mode(model_name)

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
        thinking_mode: str = "medium",
        images: Optional[list[str]] = None,
        **kwargs,
    ):
        """Generate content using the custom Gemini API host.

        Uses the same generation logic as the parent Gemini provider but
        with the custom client configured for the custom host.
        """
        # Call parent method which will use our custom client
        response = super().generate_content(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_mode=thinking_mode,
            images=images,
            **kwargs,
        )

        # Update friendly name in response to indicate custom host
        response.friendly_name = self.FRIENDLY_NAME

        return response

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens for the given text.

        Uses the same token counting logic as parent Gemini provider.
        """
        return super().count_tokens(text, model_name)

    def get_preferred_model(self, category, allowed_models):
        """Get preferred model for a given category.

        Uses the same model preference logic as parent Gemini provider
        since model characteristics should be consistent regardless of host.
        """
        return super().get_preferred_model(category, allowed_models)