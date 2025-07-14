"""
Captioning module for Event-Enriched Image Captioning
=====================================================

This module provides utilities for:
- Visual context extraction from images
- Context-aware caption generation
- Integration of visual and textual information
"""

from .visual_extractor import VisualExtractor
from .caption_generator import CaptionGenerator

__all__ = ["VisualExtractor", "CaptionGenerator"]