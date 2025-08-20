"""
Pandas Type Detector

A modular, extensible system for automatically detecting and converting pandas DataFrame column types
using a strategy pattern with confidence scoring and locale-aware parsing.
"""

from .type_detector import (
    DataType,
    DetectionResult,
    LocaleConfig,
    LOCALES,
    TypeDetector,
    NumericDetector,
    DateTimeDetector,
    BooleanDetector,
    TextDetector,
    TypeDetectionPipeline
)

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "DataType",
    "DetectionResult",
    "LocaleConfig",
    "LOCALES",
    "TypeDetector",
    "NumericDetector",
    "DateTimeDetector",
    "BooleanDetector",
    "TextDetector",
    "TypeDetectionPipeline"
]
