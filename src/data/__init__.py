"""
Data handling module for Event-Enriched Image Captioning
========================================================

This module provides utilities for:
- Downloading challenge datasets from Google Drive
- Preprocessing and organizing data
- Data validation and integrity checks
"""

from .downloader import DataDownloader

__all__ = ["DataDownloader"]