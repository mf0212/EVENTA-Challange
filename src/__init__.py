"""
Event-Enriched Image Retrieval and Captioning
============================================

A comprehensive solution for generating rich, contextually-aware image captions
by combining multimodal retrieval with advanced language models.

This package provides:
- Data downloading and preprocessing utilities
- Retrieval module for finding relevant contextual information
- Captioning module for generating enhanced descriptions
- Model implementations and utilities
"""

__version__ = "1.0.0"
__author__ = "Challenge Team"
__email__ = "team@example.com"

from .data import downloader
from .captioning import visual_extractor, caption_generator
from .models import qwen_models
from .utils import logging_utils

__all__ = [
    "downloader",
    "visual_extractor", 
    "caption_generator",
    "qwen_models",
    "logging_utils"
]