import json
import logging

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.errors import RateLimitError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Use a fast and capable model for reranking judgments
DEFAULT_GEMINI_RERANK_MODEL = "gemini-1.5-flash-latest"


class RelevanceResponse(BaseModel):
    """Pydantic model for the expected JSON response from Gemini for relevance."""

    is_relevant: bool = Field(..., description="True if the passage is relevant to the query, False otherwise.")


class GeminiRerankerClient(CrossEncoderClient):
    """
    Reranks passages using a Google Gemini model by asking it to classify relevance.

    This client makes concurrent calls to a Gemini model for each passage, asking
    if it's relevant to the query using a JSON output format. Passages marked
    as relevant are scored higher (1.0) than those marked irrelevant (0.0).
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        # client is not used directly, SDK configured globally/per call needs
    ):
        """
        Initialize the GeminiRerankerClient.

        Args:
            config (LLMConfig | None): Configuration for the underlying Gemini model
                                       (API key, model name, etc.). Temperature is
                                       forced to 0 for deterministic relevance checks.
        """
        if config is None:
            config = LLMConfig()

        # Force temperature to 0 for consistent relevance judgments
        config.temperature = 0
        # Set a reasonable max_tokens for a boolean JSON response
        config.max_tokens = 50  # Should be plenty for {"is_relevant": true/false}

        self.config = config

        # Configure SDK (safe to call multiple times)
        try:
            genai.configure(api_key=config.api_key)
        except Exception:
            logger.exception("Failed to configure Google Generative AI SDK")
            raise

        self.model_name = config.model or DEFAULT_GEMINI_RERANK_MODEL
        self.generation_config = genai.types.GenerationConfig(
            max_output_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            response_mime_type="application/json",  # Ensure JSON output
        )
        # We create a model instance per request in rank() for concurrency,
        # or could potentially share one if thread-safety/async handling allows.
        # For simplicity with semaphore_gather, we'll pass config to individual calls.

    async def _get_relevance_score(self, query: str, passage: str) -> float:
        """Internal method to call Gemini API for a single passage's relevance."""
        system_prompt = (
            "You are an expert relevance judge. Determine if the provided PASSAGE "
            "is relevant to the given QUERY. Respond ONLY with a JSON object "
            f"matching this schema: {json.dumps(RelevanceResponse.model_json_schema())}"
        )
        user_prompt = f"""
        Evaluate the relevance of the following PASSAGE to the QUERY.

        <QUERY>
        {query}
        </QUERY>

        <PASSAGE>
        {passage}
        </PASSAGE>

        Respond ONLY with the specified JSON format.
        """
        response = None
        try:
            # Create model instance for this specific call
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                system_instruction=system_prompt,
            )
            response = await model.generate_content_async(user_prompt)

            # Parse the JSON response using the Pydantic model
            response_data = json.loads(response.text)
            relevance = RelevanceResponse(**response_data)

        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"Gemini API rate limit hit during reranking: {e}")
            # Re-raise specific error for semaphore_gather or outer handler
            raise RateLimitError from e
        except (json.JSONDecodeError, TypeError, ValueError):  # Catch Pydantic validation errors too
            logger.exception(
                f"Failed to parse or validate Gemini relevance response. "
                f"Response: '{response.text if response else response}'"
            )
            return 0.0  # Score as irrelevant if response is invalid
        except Exception:
            logger.exception("Unexpected error getting relevance score")
            return 0.0  # Score as irrelevant on unexpected errors
        return 1.0 if relevance.is_relevant else 0.0

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Rank the given passages based on their relevance to the query using Gemini.

        Args:
            query (str): The query string.
            passages (list[str]): A list of passages to rank.

        Returns:
            list[tuple[str, float]]: A list of tuples containing the passage and its
                                     relevance score (1.0 for relevant, 0.0 for not),
                                     sorted in descending order of relevance.
        """
        if not passages:
            return []

        # Create tasks for getting relevance scores concurrently
        tasks = [self._get_relevance_score(query, passage) for passage in passages]

        try:
            # Use semaphore_gather to run tasks concurrently with potential rate limiting
            scores = await semaphore_gather(*tasks)
        except RateLimitError as e:
            # Propagate RateLimitError if semaphore_gather fails due to it
            logger.exception("Gemini reranking failed due to rate limits")
            raise RateLimitError from e
        except Exception:
            logger.exception("Error during concurrent Gemini reranking calls")
            raise  # Re-raise other unexpected errors from semaphore_gather

        # Combine passages with their scores
        # Ensure scores list length matches passages list length
        if len(scores) != len(passages):
            logger.error(
                f"Mismatch between number of passages ({len(passages)}) and scores received ({len(scores)}). "
                "This might indicate partial failure in concurrent calls."
            )
            # Handle mismatch: Option 1: Raise error.
            # Option 2: Pad scores (e.g., with 0.0).
            # Option 3: Use available scores.
            # Let's use available scores but log a warning.
            min_len = min(len(passages), len(scores))
            results = [(passages[i], float(scores[i])) for i in range(min_len)]
        else:
            results = [(passage, float(score)) for passage, score in zip(passages, scores, strict=False)]

        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results
