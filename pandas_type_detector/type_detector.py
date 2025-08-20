"""
Type Detection System for DataFrame Columns

A modular, extensible system for automatically detecting and converting DataFrame column types
using a strategy pattern with confidence scoring.
"""

import logging
import pandas as pd
import re

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List


class DataType(Enum):
    """Supported data types for conversion."""
    INTEGER = "integer"
    FLOAT = "float"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class DetectionResult:
    """Result of type detection with confidence score."""
    data_type: DataType
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any]  # Additional info like format, locale, etc.

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


class LocaleConfig:
    """Configuration for locale-specific formatting rules."""

    def __init__(self,
                 name: str,
                 decimal_separator: str = ".",
                 thousands_separator: str = ",",
                 currency_symbols: List[str] = [],
                 date_formats: List[str] = []):
        self.name = name
        self.decimal_separator = decimal_separator
        self.thousands_separator = thousands_separator
        self.currency_symbols = currency_symbols
        self.date_formats = date_formats


# Predefined locale configurations
LOCALES = {
    "en-us": LocaleConfig(
        name="en-us",
        decimal_separator=".",
        thousands_separator=",",
        currency_symbols=["$", "USD"],
        date_formats=[
            r'^\d{1,2}/\d{1,2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{4}-\d{2}-\d{2}$',      # YYYY-MM-DD
        ]
    ),
    "pt-br": LocaleConfig(
        name="pt-br",
        decimal_separator=",",
        thousands_separator=".",
        currency_symbols=["R$", "BRL"],
        date_formats=[
            r'^\d{1,2}/\d{1,2}/\d{4}$',  # DD/MM/YYYY
            r'^\d{4}-\d{2}-\d{2}$',      # YYYY-MM-DD
        ]
    )
}


class TypeDetector(ABC):
    """Abstract base class for type detectors."""

    def __init__(self, locale_config: LocaleConfig, sample_size: int = 1000):
        self.locale = locale_config
        self.sample_size = sample_size

    @abstractmethod
    def detect(self, series: pd.Series) -> DetectionResult:
        """
        Detect the data type of a pandas Series.

        Args:
            series: The pandas Series to analyze

        Returns:
            DetectionResult with confidence score and metadata
        """
        pass

    @abstractmethod
    def convert(self, series: pd.Series) -> pd.Series:
        """
        Convert a pandas Series to the detected type.

        Args:
            series: The pandas Series to convert

        Returns:
            Converted pandas Series
        """
        pass

    def _get_sample(self, series: pd.Series) -> pd.Series:
        """Get a representative sample of non-null values."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return non_null
        return non_null.head(min(self.sample_size, len(non_null)))


class NumericDetector(TypeDetector):
    """Detector for numeric data (integers and floats)."""

    def __init__(self, locale_config: LocaleConfig, sample_size: int = 1000,
                 min_confidence: float = 0.8, on_error: str = "coerce"):
        super().__init__(locale_config, sample_size)
        self.min_confidence = min_confidence
        self.on_error = on_error

    def detect(self, series: pd.Series) -> DetectionResult:
        """Detect numeric data with locale-aware parsing."""
        sample = self._get_sample(series)

        if len(sample) == 0:
            return DetectionResult(DataType.UNKNOWN, 0.0, {})

        # Count successful numeric conversions
        numeric_count = 0
        placeholder_count = 0
        is_integer_type = True

        for value in sample:
            if self._is_placeholder(value):
                placeholder_count += 1
                continue

            normalized = self._normalize_numeric_string(str(value))
            if normalized is not None:
                try:
                    float_val = float(normalized)
                    numeric_count += 1
                    # Check if it has decimal places
                    if float_val != int(float_val):
                        is_integer_type = False
                except ValueError:
                    pass

        total_valid = numeric_count + placeholder_count
        confidence = total_valid / len(sample) if len(sample) > 0 else 0.0

        if confidence >= self.min_confidence:
            data_type = DataType.INTEGER if is_integer_type else DataType.FLOAT
            metadata = {
                "locale": self.locale.name,
                "is_integer": is_integer_type,
                "valid_count": numeric_count,
                "placeholder_count": placeholder_count
            }
            return DetectionResult(data_type, confidence, metadata)

        return DetectionResult(DataType.UNKNOWN, confidence, {})

    def convert(self, series: pd.Series) -> pd.Series:
        """Convert series to numeric type with error handling."""
        normalized = series.astype(str).apply(self._normalize_numeric_string)
        errors_opt = 'coerce' if self.on_error == 'coerce' else 'raise'
        numeric_series = pd.to_numeric(normalized, errors=errors_opt)
        failed_mask = numeric_series.isna() & series.notna()
        error_values = series[failed_mask].unique().tolist() if failed_mask.any() else []
        if error_values:
            logging.debug(f"Numeric conversion errors: {error_values}")
        detection_result = self.detect(series)
        if detection_result.metadata.get("is_integer", False) and numeric_series.notna().any():
            return numeric_series.astype("Int64")
        else:
            return numeric_series.astype("float64")

    def _is_placeholder(self, value: Any) -> bool:
        """Check if value is a placeholder for missing data."""
        if pd.isna(value):
            return True
        str_val = str(value).strip()
        return str_val in ["", "-", "--", "---", "n/a", "N/A", "NaN"]

    def _normalize_numeric_string(self, value: str) -> Optional[str]:
        """Normalize numeric string based on locale."""
        if pd.isna(value) or self._is_placeholder(value):
            return None

        value = str(value).strip()

        # Remove currency symbols
        for symbol in self.locale.currency_symbols:
            value = value.replace(symbol, "")

        # Remove extra whitespace
        value = re.sub(r'\s+', '', value)

        if self.locale.name == "pt-br":
            return self._normalize_ptbr_numeric(value)
        elif self.locale.name == "en-us":
            return self._normalize_enus_numeric(value)
        else:
            # Default to en-us behavior
            return self._normalize_enus_numeric(value)

    def _normalize_ptbr_numeric(self, value: str) -> Optional[str]:
        """Normalize PT-BR format: 1.234.567,89"""
        # Remove any non-digit, non-comma, non-dot characters
        value = re.sub(r'[^\d,.-]', '', value)

        if not value or value in ["-", "--", "---"]:
            return None

        # Handle PT-BR format: thousands with dots, decimals with comma
        if "," in value and "." in value:
            # Both present - should be PT-BR format
            comma_pos = value.rfind(",")
            if comma_pos == len(value) - 3 or comma_pos == len(value) - 2:  # ,XX or ,X
                # Comma is decimal separator
                integer_part = value[:comma_pos].replace(".", "")
                decimal_part = value[comma_pos + 1:]
                value = f"{integer_part}.{decimal_part}"
            else:
                return None  # Invalid format
        elif "," in value:
            # Only comma - likely decimal separator in PT-BR
            if value.count(",") == 1:
                value = value.replace(",", ".")
            else:
                return None  # Multiple commas
        elif "." in value:
            # Only dots - could be thousands or decimal
            dot_count = value.count(".")
            if dot_count == 1:
                # Single dot - ambiguous, check if it looks like decimal
                if re.match(r'^\d+\.\d{1,2}$', value):
                    # Looks like decimal (1-2 digits after dot)
                    pass
                else:
                    # Probably thousands separator
                    value = value.replace(".", "")
            else:
                # Multiple dots - thousands separators
                value = value.replace(".", "")

        # Final validation
        try:
            float(value)
            return value
        except ValueError:
            return None

    def _normalize_enus_numeric(self, value: str) -> Optional[str]:
        """Normalize EN-US format: 1,234,567.89"""
        # Remove any non-digit, non-comma, non-dot characters
        value = re.sub(r'[^\d,.-]', '', value)

        if not value or value in ["-", "--", "---"]:
            return None

        # Remove thousands separators (commas)
        value = value.replace(",", "")

        # Validate final result
        try:
            float(value)
            return value
        except ValueError:
            return None


class DateTimeDetector(TypeDetector):
    """Detector for datetime data."""

    def __init__(self, locale_config: LocaleConfig, sample_size: int = 1000,
                 min_confidence: float = 0.7, on_error: str = "coerce"):
        super().__init__(locale_config, sample_size)
        self.min_confidence = min_confidence
        self.on_error = on_error

    def detect(self, series: pd.Series) -> DetectionResult:
        """Detect datetime data."""
        sample = self._get_sample(series)

        if len(sample) == 0:
            return DetectionResult(DataType.UNKNOWN, 0.0, {})

        # First check: do values look like PT-BR numbers? If so, probably not dates
        if self.locale.name == "pt-br":
            numeric_detector = NumericDetector(self.locale, self.sample_size)
            numeric_result = numeric_detector.detect(series)
            if numeric_result.confidence > 0.6:
                # Probably numeric, not dates
                return DetectionResult(DataType.UNKNOWN, 0.0,
                                       {"rejected_reason": "looks_like_numeric"})

        # Try pandas datetime parsing
        try:
            converted = pd.to_datetime(sample, errors='coerce', format='mixed')
            success_count = converted.notna().sum()
            confidence = success_count / len(sample)

            if confidence >= self.min_confidence:
                metadata = {
                    "locale": self.locale.name,
                    "success_count": success_count,
                    "parsing_method": "pandas_mixed"
                }
                return DetectionResult(DataType.DATETIME, confidence, metadata)
        except Exception as e:
            logging.debug(f"Pandas datetime parsing failed: {e}")

        # Try pattern matching
        pattern_confidence = self._check_date_patterns(sample)
        if pattern_confidence >= self.min_confidence:
            metadata = {
                "locale": self.locale.name,
                "parsing_method": "pattern_matching",
                "pattern_confidence": pattern_confidence
            }
            return DetectionResult(DataType.DATETIME, pattern_confidence, metadata)

        return DetectionResult(DataType.UNKNOWN, max(confidence, pattern_confidence), {})

    def convert(self, series: pd.Series) -> pd.Series:
        """Convert series to datetime with error handling."""
        errors_opt = 'coerce' if self.on_error == 'coerce' else 'raise'
        try:
            dt_series = pd.to_datetime(series, errors=errors_opt, format='mixed')
            failed_mask = dt_series.isna() & series.notna()
            error_values = series[failed_mask].unique().tolist() if failed_mask.any() else []
            if error_values:
                logging.debug(f"Datetime conversion errors: {error_values}")
            return dt_series
        except Exception as e:
            logging.debug(f"Datetime conversion exception: {e}")
            return series

    def _check_date_patterns(self, sample: pd.Series) -> float:
        """Check if sample matches common date patterns."""
        if len(sample) == 0:
            return 0.0

        # Common date patterns
        patterns = [
            r'^\d{4}-\d{2}-\d{2}$',                      # YYYY-MM-DD
            r'^\d{4}/\d{2}/\d{2}$',                      # YYYY/MM/DD
            r'^\d{1,2}/\d{1,2}/\d{4}$',                  # MM/DD/YYYY or DD/MM/YYYY
            r'^\d{1,2}-\d{1,2}-\d{4}$',                  # MM-DD-YYYY or DD-MM-YYYY
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',     # ISO datetime
            r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',     # YYYY-MM-DD HH:MM:SS
        ]

        matches = 0
        for value in sample:
            if isinstance(value, str):
                value_clean = str(value).strip()
                if any(re.match(pattern, value_clean) for pattern in patterns):
                    matches += 1

        return matches / len(sample)


class BooleanDetector(TypeDetector):
    """Detector for boolean data."""

    def __init__(self, locale_config: LocaleConfig, sample_size: int = 1000,
                 min_confidence: float = 0.8, on_error: str = "coerce"):
        super().__init__(locale_config, sample_size)
        self.min_confidence = min_confidence
        self.on_error = on_error

        # Define boolean indicators for different locales
        # Note: We exclude "1" and "0" to avoid conflicts with numeric data
        if locale_config.name == "pt-br":
            self.true_values = {"sim", "verdadeiro", "true", "s", "v"}
            self.false_values = {"não", "nao", "falso", "false", "n", "f"}
        else:  # Default to English
            self.true_values = {"true", "yes", "y", "t"}
            self.false_values = {"false", "no", "n", "f"}

    def detect(self, series: pd.Series) -> DetectionResult:
        """Detect boolean data."""
        sample = self._get_sample(series)

        if len(sample) == 0:
            return DetectionResult(DataType.UNKNOWN, 0.0, {})

        boolean_count = 0
        for value in sample:
            if self._is_boolean_value(value):
                boolean_count += 1

        confidence = boolean_count / len(sample)

        if confidence >= self.min_confidence:
            metadata = {"locale": self.locale.name, "boolean_count": boolean_count}
            return DetectionResult(DataType.BOOLEAN, confidence, metadata)

        return DetectionResult(DataType.UNKNOWN, confidence, {})

    def convert(self, series: pd.Series) -> pd.Series:
        """Convert series to boolean with error handling."""
        mapped = series.astype(str).str.lower().str.strip().map(
            lambda x: True if x in self.true_values
            else False if x in self.false_values
            else None
        )
        bool_series = mapped.astype("boolean")
        error_values = series[mapped.isna()].unique().tolist() if (
            self.on_error == 'coerce' and mapped.isna().any()) else []
        if error_values:
            logging.debug(f"Boolean conversion errors: {error_values}")
        return bool_series

    def _is_boolean_value(self, value: Any) -> bool:
        """Check if value represents a boolean."""
        if pd.isna(value):
            return False
        str_val = str(value).lower().strip()
        return str_val in self.true_values or str_val in self.false_values


class TextDetector(TypeDetector):
    """Detector for text data (fallback)."""

    def detect(self, series: pd.Series) -> DetectionResult:
        """Text is the fallback type - always has some confidence."""
        metadata = {"locale": self.locale.name, "fallback": True}
        return DetectionResult(DataType.TEXT, 0.1, metadata)  # Low confidence fallback

    def convert(self, series: pd.Series) -> pd.Series:
        """Convert series to string type (no error handling needed)."""
        return series.astype(str)


class TypeDetectionPipeline:
    """Main pipeline for detecting and converting DataFrame column types."""

    def __init__(self, locale: str = "en-us", sample_size: int = 1000,
                 on_error: str = "coerce"):
        """
        Initialize the type detection pipeline.

        Args:
            locale: Locale for formatting rules ("en-us", "pt-br", etc.)
            sample_size: Number of values to sample for type detection
            on_error: Error handling strategy - "coerce" (convert to NaN),
                     "raise" (raise exceptions), or "ignore" (keep original)
        """
        if locale not in LOCALES:
            raise ValueError(f"Unsupported locale: {locale}. "
                             f"Supported locales: {list(LOCALES.keys())}")

        if on_error not in ["coerce", "raise", "ignore"]:
            raise ValueError(f"Invalid on_error value: {on_error}. "
                             f"Must be one of: 'coerce', 'raise', 'ignore'")

        self.locale_config = LOCALES[locale]
        self.sample_size = sample_size
        self.on_error = on_error

        # Initialize detectors in order of specificity (most specific first)
        self.detectors = [
            BooleanDetector(self.locale_config, sample_size, on_error=self.on_error),
            DateTimeDetector(self.locale_config, sample_size, on_error=self.on_error),
            NumericDetector(self.locale_config, sample_size, on_error=self.on_error),
            TextDetector(self.locale_config, sample_size),  # Fallback
        ]

    def detect_column_type(self, series: pd.Series) -> DetectionResult:
        """
        Detect the best data type for a column.

        Args:
            series: The pandas Series to analyze

        Returns:
            DetectionResult with the best type detected
        """
        best_result = DetectionResult(DataType.UNKNOWN, 0.0, {})

        for detector in self.detectors:
            result = detector.detect(series)
            if result.confidence > best_result.confidence:
                best_result = result
                # If we get high confidence, we can stop
                if result.confidence >= 0.9:
                    break

        return best_result

    def fix_dataframe_dtypes(self, df: pd.DataFrame,
                             skip_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detect and convert all columns in a DataFrame to optimal data types.

        Args:
            df: Input DataFrame
            skip_columns: List of column names to skip during conversion

        Returns:
            DataFrame with optimized data types
        """

        df = df.copy()
        skip_columns = skip_columns or []

        for column in df.columns:
            if column in skip_columns:
                continue

            if df[column].empty:
                continue

            old_dtype = df[column].dtype
            # Detect type
            result = self.detect_column_type(df[column])

            if result.data_type != DataType.UNKNOWN:
                try:
                    # Find the appropriate detector and convert
                    for detector in self.detectors:
                        test_result = detector.detect(df[column])
                        if (test_result.data_type == result.data_type and
                                test_result.confidence == result.confidence):
                            df[column] = detector.convert(df[column])
                            new_dtype = df[column].dtype
                            logging.info(
                                f"{column:<15} {str(old_dtype):<10} {str(new_dtype):<10} {result.confidence:<4.2f}"
                            )
                            break
                except Exception as e:
                    if self.on_error == "raise":
                        raise RuntimeError(f"Failed to convert column {column}: {e}") from e
                    # For 'coerce' and 'ignore', skip logging

        return df
