"""
Utility functions for Event-Enriched Image Captioning
=====================================================

This module provides:
- Logging configuration and utilities
- Evaluation metrics and scoring
- Common helper functions
"""

from .logging_utils import setup_logging, get_logger
from .evaluation import calculate_cider, calculate_bleu

__all__ = ["setup_logging", "get_logger", "calculate_cider", "calculate_bleu"]