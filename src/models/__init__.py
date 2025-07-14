"""
Model implementations for Event-Enriched Image Captioning
=========================================================

This module provides:
- Qwen-VL model implementations
- Model loading and configuration utilities
- Model wrapper classes for easy usage
"""

from .qwen_models import QwenVLModel, QwenTextModel
from .embedding_models import EmbeddingModel

__all__ = ["QwenVLModel", "QwenTextModel", "EmbeddingModel"]