"""
Embedding model implementations for Event-Enriched Image Captioning
=============================================================

This module provides wrapper classes for BGE-M3 embedding model.
"""

from typing import Optional, Dict, Any, List
from ..utils.logging_utils import get_logger, LoggerMixin
from FlagEmbedding import BGEM3FlagModel

logger = get_logger(__name__)

class EmbeddingModel(LoggerMixin):
    """
    Base class for embedding models.
    """
    
    def __init__(self):
        """
        Initialize embedding model.
        """
        self.model = None
    
    def load_model(self) -> None:
        """
        Load the embedding model.
        """
        self.model = BGEM3FlagModel('BAAI/bge-m3')

    def embed(self, texts: List[str], max_len: int) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of input texts
            max_len: Maximum length of the input texts

        Returns:
            List of embeddings
        """
        if not self.model:
            self.logger.error("Model not loaded")
            return []

        try:
            embeddings = self.model.encode(texts,
                                           batch_size=12,
                                           max_length=(max_len // 512 + 1) * 512,
                                           return_dense=True,
                                           return_sparse=False,
                                           return_colbert_vecs=False)
            return embeddings['dense_vecs']
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            return []