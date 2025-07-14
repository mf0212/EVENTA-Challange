"""
Retrieval module for Event-Enriched Image Captioning
====================================================

This module provides utilities for:
- Multimodal search and retrieval from article databases
- Content-visual and visual-visual similarity matching
- Context extraction and refinement
"""

from .retriever import Retriever

__all__ = ["Retriever"]