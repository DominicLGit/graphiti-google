import logging
import typing
from collections.abc import Iterable
from typing import Literal

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
from graphiti_core.llm_client.errors import RateLimitError

logger = logging.getLogger(__name__)

# Common embedding models: 'embedding-001', 'text-embedding-004'
# See https://ai.google.dev/models/gemini#embedding_models
DEFAULT_GEMINI_EMBEDDING_MODEL = "text-embedding-004"  # 768 dimensions

# Map model names to their dimensions (add more as needed)
GEMINI_EMBEDDING_DIMS = {
    "embedding-001": 768,
    "text-embedding-004": 768,
}
DEFAULT_EMBEDDING_DIM = GEMINI_EMBEDDING_DIMS[DEFAULT_GEMINI_EMBEDDING_MODEL]

# Supported task types for embedding models
GeminiEmbeddingTaskType = Literal[
    "TASK_TYPE_UNSPECIFIED",
    "RETRIEVAL_QUERY",  # Query embedding for retrieval
    "RETRIEVAL_DOCUMENT",  # Document embedding for retrieval
    "SEMANTIC_SIMILARITY",  # General semantic similarity
    "CLASSIFICATION",
    "CLUSTERING",
]
DEFAULT_TASK_TYPE: GeminiEmbeddingTaskType = "RETRIEVAL_DOCUMENT"


class NoEmbeddingFieldError(Exception):
    """Exception raised when the API response does not contain an 'embedding' field."""

    def __init__(self, message="The API response does not contain an 'embedding' field."):
        self.message = message
        super().__init__(self.message)


class UnsupportedInputTypeError(Exception):
    """Exception raised when the input type is not supported for embedding."""

    def __init__(self, message="Unsupported input type for embedding.", input_type=None):
        self.message = message + f" Input type: {input_type}"
        super().__init__(self.message)


class GeminiEmbedderConfig(EmbedderConfig):
    """Configuration specific to the Gemini Embedder."""

    # Override default dimension if a specific model is chosen
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    embedding_model: str = DEFAULT_GEMINI_EMBEDDING_MODEL
    task_type: GeminiEmbeddingTaskType = DEFAULT_TASK_TYPE
    api_key: str | None = None
    # base_url is not used by the google-generativeai library directly

    def __init__(self, **data):
        super().__init__(**data)
        # Automatically set embedding_dim based on model if not explicitly provided
        if "embedding_model" in data and data["embedding_model"] in GEMINI_EMBEDDING_DIMS:
            if "embedding_dim" not in data:  # Only override if not set by user
                self.embedding_dim = GEMINI_EMBEDDING_DIMS[data["embedding_model"]]
        elif self.embedding_model in GEMINI_EMBEDDING_DIMS:
            # handles case where embedding_model is default but dim isn't
            if "embedding_dim" not in data:
                self.embedding_dim = GEMINI_EMBEDDING_DIMS[self.embedding_model]
        else:
            logger.warning(
                f"Unknown Gemini embedding model '{self.embedding_model}'. "
                f"Using default dimension {self.embedding_dim}. "
                f"Ensure this matches the actual model output dimension."
            )


class GeminiEmbedderClient(EmbedderClient):
    """
    Google Gemini Embedder Client.

    Uses the google.generativeai library to create text embeddings.
    """

    def __init__(
        self,
        config: GeminiEmbedderConfig | None = None,
    ):
        if config is None:
            config = GeminiEmbedderConfig()
        self.config = config

        try:
            genai.configure(api_key=config.api_key)
        except Exception:
            logger.exception("Failed to configure Google Generative AI SDK")
            raise

        valid_task_types = typing.get_args(GeminiEmbeddingTaskType)
        if self.config.task_type not in valid_task_types:
            logger.warning(
                f"Invalid task_type '{self.config.task_type}'. "
                f"Defaulting to '{DEFAULT_TASK_TYPE}'. Valid types: {valid_task_types}"
            )
            self.config.task_type = DEFAULT_TASK_TYPE

    async def create(self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]) -> list[float]:
        """
        Creates an embedding for the given input data.

        Args:
            input_data: A string, list of strings. Iterables of ints are not
                        directly supported by the Gemini API for text embedding
                        and will raise a TypeError.

        Returns:
            A list of floats representing the embedding. If input is a list of strings,
            it currently returns the embedding for the *first* string, matching the
            OpenAIEmbedder behavior. Consider using batch embedding methods if needed.

        Raises:
            TypeError: If input_data is an iterable of integers.
            RateLimitError: If the API rate limit is exceeded.
            Exception: For other API errors.
        """
        if isinstance(input_data, str):
            content_to_embed = input_data
        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            # Match OpenAIEmbedder behavior: embed only the first item if a list is given.
            # For batching, genai.embed_content supports list input directly.
            # Modify this if batch behavior is desired.
            if not input_data:
                return []  # Handle empty list case
            content_to_embed = input_data[0]
            if len(input_data) > 1:
                logger.warning("Input is a list of strings, but only embedding the first item.")
        elif isinstance(input_data, Iterable):
            # Gemini text embedding models do not accept token IDs directly via this API.
            raise UnsupportedInputTypeError(input_type=type(input_data))
        else:
            raise UnsupportedInputTypeError(input_type=type(input_data))

        # Gemini model name needs 'models/' prefix for embedding API
        model_name = f"models/{self.config.embedding_model}"

        try:
            result = await genai.embed_content_async(
                model=model_name,
                content=content_to_embed,
                task_type=self.config.task_type,
                # title= optional title for RETRIEVAL_DOCUMENT
            )

            # Do NOT truncate by default, return the actual embedding from the model.

        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"Gemini API rate limit exceeded: {e}")
            raise RateLimitError from e
        except Exception:
            logger.exception("Error creating Gemini embedding")
            raise

        # The result dictionary contains {'embedding': [float_values]}
        embedding = result.get("embedding")
        if embedding is None:
            logger.error(f"Gemini embedding API did not return an 'embedding' field. Result: {result}")
            raise NoEmbeddingFieldError

        # Check dimension consistency (optional but recommended)
        if len(embedding) != self.config.embedding_dim:
            logger.warning(
                f"Model '{self.config.embedding_model}' returned dimension {len(embedding)}, "
                f"but configured dimension is {self.config.embedding_dim}. Returning actual dimension."
            )
        return embedding
