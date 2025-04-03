import json
import typing

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from graphiti_core.llm_client.client import DEFAULT_MAX_TOKENS, LLMClient, LLMConfig
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.prompts.models import Message
from loguru import logger
from pydantic import BaseModel

# See https://ai.google.dev/models/gemini for available models
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-lite"


class EmptyResponseError(Exception):
    """Exception raised when the API returns an empty response."""

    def __init__(self, message="Received empty response from Gemini API"):
        self.message = message
        super().__init__(self.message)


class GeminiClient(LLMClient):
    """
    Google Gemini Client for interacting with Gemini language models.

    This class extends the LLMClient and provides methods to initialize the client
    and generate responses from the Gemini model, enforcing JSON output.

    Attributes:
        client (genai.GenerativeModel): The Gemini client model instance.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.
        generation_config (genai.types.GenerationConfig): Configuration for generation.
    """

    def __init__(self, config: LLMConfig | None = None, *, cache: bool = False):
        """
        Initialize the GeminiClient with the provided configuration and cache setting.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key,
                                       model, base URL (ignored for Gemini), temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
        """
        if config is None:
            config = LLMConfig()
        # Ensure max_tokens has a default if not set
        if config.max_tokens is None:
            config.max_tokens = DEFAULT_MAX_TOKENS

        super().__init__(config, cache)

        # Configure the Google AI SDK
        # The SDK will attempt to find the API key from the environment variable GOOGLE_API_KEY if not provided
        try:
            genai.configure(api_key=config.api_key)
        except Exception as e:
            logger.exception(f"Failed to configure Google Generative AI SDK: {e}")
            raise  # Re-raise exception if configuration fails

        self.model = config.model or DEFAULT_GEMINI_MODEL
        self.generation_config = genai.types.GenerationConfig(
            # Gemini uses 'max_output_tokens'
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            # Force JSON output
            response_mime_type="application/json",
        )
        self.gemini_model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=self.generation_config,
        )

    async def _generate_response(  # noqa: C901
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,  # Passed but Gemini config is set in init
    ) -> dict[str, typing.Any]:
        """
        Generates a response from the Gemini model based on the provided messages.

        Args:
            messages (list[Message]): The list of messages forming the conversation history.
                                     The last message contains the main prompt.
            response_model (type[BaseModel] | None): Pydantic model for expected JSON structure.
                                                    Schema added to the prompt.
            max_tokens (int): Maximum tokens for the response (Note: controlled by generation_config).

        Returns:
            dict[str, typing.Any]: The parsed JSON response from the model.

        Raises:
            RateLimitError: If the API rate limit is exceeded.
            Exception: For other API or parsing errors.
        """
        system_instruction = None
        gemini_formatted_messages: list[genai.types.ContentDict] = []

        # Separate system instruction if present (Gemini handles it specifically)
        if messages and messages[0].role == "system":
            system_instruction = messages[0].content
            messages_to_process = messages[1:]
        else:
            messages_to_process = messages

        # Convert messages to Gemini's format
        for m in messages_to_process:
            # Gemini alternates roles between 'user' and 'model' ('assistant' maps to 'model')
            role = "user" if m.role == "user" else "model"
            cleaned_content = self._clean_input(m.content)  # Clean content before sending
            gemini_formatted_messages.append({"role": role, "parts": [{"text": cleaned_content}]})

        # Append response model schema if provided
        if response_model is not None:
            serialized_model = json.dumps(response_model.model_json_schema())
            schema_prompt = f"\n\nRespond ONLY with a JSON object matching this schema:\n\n{serialized_model}"
            # Add schema instructions to the last message's content
            if gemini_formatted_messages:
                last_message_content = gemini_formatted_messages[-1]["parts"][0]["text"]
                gemini_formatted_messages[-1]["parts"][0]["text"] = last_message_content + schema_prompt
            elif system_instruction:  # If only a system prompt was given, add schema there
                system_instruction += schema_prompt
            else:  # Should not happen with valid input, but handle edge case
                gemini_formatted_messages.append({"role": "user", "parts": [{"text": schema_prompt}]})

        try:
            # Use the client configured with JSON output
            response = await self.gemini_model.generate_content_async(
                contents=gemini_formatted_messages,
            )
        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"Gemini API rate limit exceeded: {e}")
            raise RateLimitError from e
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            raise

        response_text = response.text
        if not response_text:
            logger.error("Gemini returned an empty response.")
            raise EmptyResponseError

        try:
            return json.loads(response_text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON response from Gemini: {response_text} \n Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            raise
