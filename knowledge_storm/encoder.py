import os
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Union, Optional
from pathlib import Path

try:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        import litellm

        litellm.drop_params = True
        litellm.telemetry = False

    from litellm.caching.caching import Cache

    disk_cache_dir = os.path.join(Path.home(), ".storm_local_cache")
    litellm.cache = Cache(disk_cache_dir=disk_cache_dir, type="disk")

except ImportError:

    class LitellmPlaceholder:
        def __getattr__(self, _):
            raise ImportError(
                "The LiteLLM package is not installed. Run `pip install litellm`."
            )

    litellm = LitellmPlaceholder()


class Encoder:
    """
    A wrapper class for the LiteLLM embedding model, designed to handle embedding
    generation tasks efficiently. It supports parallel processing and local caching of
    embedding results for improved performance.

    The Encoder utilizes the LiteLLM library to interact with various embedding models,
    such as OpenAI and Azure embeddings. Users can specify the desired encoder type and
    provide relevant API credentials during initialization.

    Features:
        - Support for multiple embedding models (e.g., OpenAI, Azure).
        - Parallel processing for faster embedding generation.
        - Local disk caching to store and reuse embedding results.
        - Total token usage tracking for cost monitoring.

    Note:
        Refer to the LiteLLM documentation for details on supported embedding models:
        https://docs.litellm.ai/docs/embedding/supported_embedding
    """

    def __init__(
        self,
        encoder_type: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        """
        Initializes the Encoder with the appropriate embedding model.

        Args:
            encoder_type (Optional[str]): Type of encoder ('openai', 'azure', etc.).
            api_key (Optional[str]): API key for the encoder service.
            api_base (Optional[str]): API base URL for the encoder service.
            api_version (Optional[str]): API version for the encoder service.
        """
        self.embedding_model_name = None
        self.kargs = {}
        self.total_token_usage = 0

        # Initialize the appropriate embedding model
        encoder_type = encoder_type or os.getenv("ENCODER_API_TYPE")
        if not encoder_type:
            raise ValueError("ENCODER_API_TYPE environment variable is not set.")

        if encoder_type.lower() == "openai":
            self.embedding_model_name = "text-embedding-3-small"
            self.kargs = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
        elif encoder_type.lower() == "azure":
            self.embedding_model_name = "azure/text-embedding-3-small"
            self.kargs = {
                "api_key": api_key or os.getenv("AZURE_API_KEY"),
                "api_base": api_base or os.getenv("AZURE_API_BASE"),
                "api_version": api_version or os.getenv("AZURE_API_VERSION"),
            }
        else:
            raise ValueError(
                f"Unsupported ENCODER_API_TYPE '{encoder_type}'. Supported types are 'openai', 'azure', 'together'."
            )

    def get_total_token_usage(self, reset: bool = False) -> int:
        """
        Retrieves the total token usage.

        Args:
            reset (bool): If True, resets the total token usage counter after retrieval.

        Returns:
            int: The total number of tokens used.
        """
        token_usage = self.total_token_usage
        if reset:
            self.total_token_usage = 0
        return token_usage

    def encode(
        self,
        texts: Union[str, List[str]],
        max_workers: int = 5,
        batch_size: int = 128,
    ) -> np.ndarray:
        """
        Public method to get embeddings for the given texts.

        Args:
            texts (Union[str, List[str]]): A single text string or a list of text strings to embed.
            max_workers (int): The maximum number of concurrent embedding requests.
            batch_size (int): The maximum number of texts sent in a single request.

        Returns:
            np.ndarray: A 1D array for a single input string, a 2D array for a list.
        """
        return self._get_text_embeddings(
            texts, max_workers=max_workers, batch_size=batch_size
        )

    def _embed_batch(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """Embed a batch of texts in a single API request."""

        def get_field(item, name, default=None):
            # litellm may return dicts or pydantic-like objects depending on
            # the provider/version.
            try:
                return item[name]
            except (KeyError, TypeError):
                return getattr(item, name, default)

        response = litellm.embedding(
            model=self.embedding_model_name, input=texts, caching=True, **self.kargs
        )
        data = response.data
        indices = [get_field(item, "index") for item in data]
        if indices and all(index is not None for index in indices):
            data = [
                item for _, item in sorted(zip(indices, data), key=lambda pair: pair[0])
            ]
        embeddings = [get_field(item, "embedding") for item in data]
        if any(embedding is None for embedding in embeddings):
            raise RuntimeError(
                "Embedding API response is missing an 'embedding' field."
            )
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Embedding API returned {len(embeddings)} vectors for "
                f"{len(texts)} inputs."
            )
        token_usage = (response.get("usage") or {}).get("total_tokens", 0)
        return embeddings, token_usage

    def _get_text_embeddings(
        self,
        texts: Union[str, List[str]],
        max_workers: int = 5,
        batch_size: int = 128,
    ) -> np.ndarray:
        """
        Get text embeddings in batches.

        Identical texts are embedded only once, and results are restored to the
        original input order. Failures are raised instead of being silently
        dropped, so callers never receive a shorter array than they asked for.

        Args:
            texts (Union[str, List[str]]): A single text string or a list of text strings.
            max_workers (int): The maximum number of concurrent batch requests.
            batch_size (int): The maximum number of texts per request.

        Returns:
            np.ndarray: A 1D array for a single input string, a 2D array for a list.
        """
        is_single = isinstance(texts, str)
        texts = [texts] if is_single else list(texts)

        if not texts:
            return np.array([])

        unique_texts = list(dict.fromkeys(texts))
        batches = [
            unique_texts[start : start + batch_size]
            for start in range(0, len(unique_texts), batch_size)
        ]

        embeddings_by_text = {}
        total_tokens = 0

        with ThreadPoolExecutor(
            max_workers=max(1, min(max_workers, len(batches)))
        ) as executor:
            futures = {
                executor.submit(self._embed_batch, batch): batch for batch in batches
            }

            for future in as_completed(futures):
                batch = futures[future]
                try:
                    batch_embeddings, batch_tokens = future.result()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to embed a batch of {len(batch)} texts: {e}"
                    ) from e
                total_tokens += batch_tokens
                for text, embedding in zip(batch, batch_embeddings):
                    embeddings_by_text[text] = embedding

        self.total_token_usage += total_tokens
        ordered = np.array([embeddings_by_text[text] for text in texts])
        return ordered[0] if is_single else ordered
