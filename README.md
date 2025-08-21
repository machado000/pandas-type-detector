# pandas-type-detector

🔍 **Intelligent DataFrame Type Detection with Locale Awareness**

A robust, production-ready library for automatically detecting and converting pandas DataFrame column types with sophisticated locale-aware parsing, confidence scoring, and enhanced text filtering capabilities.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-17%2F17%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Key Features

- **🌍 Locale-Aware Parsing**: Native support for PT-BR and EN-US number formats, dates, and boolean values
- **🎯 Smart Text Filtering**: Advanced algorithms prevent text containing numbers from being misclassified as numeric
- **📊 Confidence Scoring**: Get reliability scores for each type detection decision
- **🛡️ Robust Error Handling**: Configurable strategies for handling conversion errors
- **⚡ Performance Optimized**: Intelligent sampling and early-exit strategies for large datasets
- **🧩 Modular Architecture**: Extensible design for adding new data types and locales
- **✅ Production Tested**: Successfully handles complex real-world data scenarios

## 📦 Installation

```bash
pip install pandas-type-detector
```

## 🎯 Quick Start

```python
import pandas as pd
from pandas_type_detector import TypeDetectionPipeline

# Sample data with mixed formats
data = {
    'revenue': ['1.234,56', '2.890,00', '543,21'],      # PT-BR currency format
    'quantity': ['10', '25', '8'],                       # Integers
    'active': ['Sim', 'Não', 'Sim'],                    # PT-BR booleans
    'date': ['2025-01-15', '2025-02-20', '2025-03-10'], # ISO dates
    'description': ['(31) Product A', '(45) Service B', '(12) Item C']  # Text with numbers
}

df = pd.DataFrame(data)
print("Original dtypes:")
print(df.dtypes)
# All columns are 'object' initially

# Initialize pipeline with Portuguese (Brazil) locale
pipeline = TypeDetectionPipeline(locale="pt-br", on_error="coerce")

# Automatically detect and convert types
df_converted = pipeline.fix_dataframe_dtypes(df)

print("\\nConverted dtypes:")
print(df_converted.dtypes)
# Output:
# revenue        float64    ← Correctly parsed PT-BR format
# quantity         Int64    ← Detected as integer
# active         boolean    ← Portuguese booleans converted
# date      datetime64[ns]  ← ISO dates parsed
# description       object  ← Text with numbers kept as text
```

## 🌐 Locale Support

### 🇧🇷 PT-BR (Portuguese Brazil)
- **Decimal separator**: `,` (comma) → `1.234,56` becomes `1234.56`
- **Thousands separator**: `.` (dot) → `1.000.000,00`
- **Currency symbols**: `R$`, `BRL`
- **Boolean values**: `Sim`/`Não`, `Verdadeiro`/`Falso`, `S`/`N`
- **Date formats**: `DD/MM/YYYY`, `YYYY-MM-DD`

### 🇺🇸 EN-US (English United States)  
- **Decimal separator**: `.` (dot) → `1,234.56`
- **Thousands separator**: `,` (comma) → `1,000,000.00`
- **Currency symbols**: `$`, `USD`
- **Boolean values**: `True`/`False`, `Yes`/`No`, `Y`/`N`
- **Date formats**: `MM/DD/YYYY`, `YYYY-MM-DD`

## 📚 Advanced Usage

### 🔧 Error Handling Strategies

```python
# Strategy 1: Coerce errors to NaN (default - recommended)
pipeline = TypeDetectionPipeline(locale="en-us", on_error="coerce")
df_safe = pipeline.fix_dataframe_dtypes(df)

# Strategy 2: Raise exceptions on conversion errors
pipeline = TypeDetectionPipeline(locale="en-us", on_error="raise")
try:
    df_strict = pipeline.fix_dataframe_dtypes(df)
except ValueError as e:
    print(f"Conversion error: {e}")

# Strategy 3: Ignore problematic columns
pipeline = TypeDetectionPipeline(locale="en-us", on_error="ignore")
df_conservative = pipeline.fix_dataframe_dtypes(df)
```

### 🔍 Individual Column Analysis

```python
# Get detailed detection information
result = pipeline.detect_column_type(df['revenue'])

print(f"Detected type: {result.data_type.value}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Locale: {result.metadata['locale']}")
print(f"Parsing details: {result.metadata}")

# Example output:
# Detected type: float
# Confidence: 95.00%
# Locale: pt-br
# Parsing details: {'locale': 'pt-br', 'is_integer': False, 'numeric_count': 3, ...}
```

### 🎛️ Column Selection and Skipping

```python
# Skip specific columns during conversion
df_converted = pipeline.fix_dataframe_dtypes(
    df, 
    skip_columns=['id', 'raw_text', 'keep_as_string']
)

# Skip columns remain as original 'object' type
# Other columns are automatically converted
```

### ⚙️ Performance Tuning

```python
# Optimize for large datasets
pipeline = TypeDetectionPipeline(
    locale="pt-br",
    sample_size=5000,      # Analyze up to 5000 rows per column (default: 1000)
    on_error="coerce"
)

# For smaller datasets, use full analysis
pipeline = TypeDetectionPipeline(
    locale="en-us", 
    sample_size=10000      # Effectively analyze all rows for small datasets
)
```

## 🛡️ Smart Text Filtering

One of the key improvements in this library is sophisticated text filtering that prevents common misclassification issues:

```python
# These text values are correctly identified as text, not numeric
problematic_data = pd.Series([
    "(31) Week from 28/jul to 3/aug",  # Text with numbers
    "(45) Product description",        # Text with parenthetical numbers  
    "Order #12345 - Item A",           # Mixed text and numbers
    "Section 3.1.4 Overview"          # Version numbers in text
])

result = pipeline.detect_column_type(problematic_data)
print(result.data_type)  # DataType.TEXT (correctly identified as text)
```

## 🧪 Testing

The library includes a comprehensive test suite with 17 test cases covering all functionality:

```bash
cd pandas-type-detector
poetry run pytest tests/test.py -v
```

### Test Coverage
- ✅ Numeric detection (integers, floats) for both locales
- ✅ Boolean detection in multiple languages
- ✅ DateTime parsing and conversion
- ✅ Text-with-numbers rejection algorithms
- ✅ Skip columns functionality
- ✅ Error handling strategies
- ✅ Real-world data scenarios
- ✅ Edge cases and boundary conditions

## 📊 Supported Data Types

| Data Type | Description | Example Values |
|-----------|-------------|----------------|
| **Integer** | Whole numbers | `123`, `1.000` (PT-BR), `1,000` (EN-US) |
| **Float** | Decimal numbers | `123,45` (PT-BR), `123.45` (EN-US) |
| **Boolean** | True/False values | `Sim/Não` (PT-BR), `Yes/No` (EN-US) |
| **DateTime** | Date and time | `2025-01-15`, `15/01/2025` |
| **Text** | String data | Any text, including mixed alphanumeric |
## 🔧 Extensibility

### Adding a New Locale

```python
from pandas_type_detector import LOCALES, LocaleConfig

# Add German locale
LOCALES['de-de'] = LocaleConfig(
    name='de-de',
    decimal_separator=',',
    thousands_separator='.',
    currency_symbols=['€', 'EUR'],
    date_formats=[r'^\\d{1,2}\\.\\d{1,2}\\.\\d{4}$']  # DD.MM.YYYY
)

# Use the new locale
pipeline = TypeDetectionPipeline(locale="de-de")
```

### Creating Custom Detectors

```python
from pandas_type_detector import TypeDetector, DataType, DetectionResult

class EmailDetector(TypeDetector):
    def detect(self, series):
        # Custom email detection logic
        email_pattern = r'^[\\w\\.-]+@[\\w\\.-]+\\.[\\w]+$'
        matches = series.str.match(email_pattern).sum()
        confidence = matches / len(series)
        
        if confidence >= 0.8:
            return DetectionResult(DataType.TEXT, confidence, {"format": "email"})
        return DetectionResult(DataType.UNKNOWN, confidence, {})
    
    def convert(self, series):
        # Email-specific processing if needed
        return series.astype(str)
```

## 🚀 Performance Characteristics

- **Memory Efficient**: Processes columns independently without loading entire dataset
- **Sampling Strategy**: Configurable sampling reduces processing time for large datasets
- **Early Exit**: Stops analysis when high confidence is reached (≥90%)
- **Production Ready**: Optimized for ETL pipelines and data processing workflows

### Benchmarks
- ✅ Tested with datasets up to 14,607 rows in production
- ✅ Handles complex mixed-format data reliably
- ✅ Minimal performance overhead on modern hardware

## 🤝 Contributing

We welcome contributions! The modular architecture makes it easy to:

1. **Add new locales** - Extend `LOCALES` configuration
2. **Create new detectors** - Inherit from `TypeDetector` base class
3. **Improve algorithms** - Enhance existing detection logic
4. **Add test cases** - Expand the test suite for new scenarios

### Development Setup

```bash
git clone https://github.com/yourusername/pandas-type-detector
cd pandas-type-detector
poetry install
poetry run pytest
```

## 📋 Requirements

- **Python**: 3.7+ (tested on 3.7, 3.8, 3.9, 3.10, 3.11, 3.12)
- **pandas**: ≥1.0.0
- **numpy**: ≥1.19.0

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This library was developed to solve real-world data quality challenges in Brazilian financial and business data processing. It has been successfully deployed in production environments handling complex PT-BR formatted datasets.

Special thanks to the pandas and NumPy communities for providing the foundation that makes this work possible.

## 📞 Support

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/yourusername/pandas-type-detector/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/yourusername/pandas-type-detector/discussions)
- **📖 Documentation**: This README and inline code documentation
- **🧪 Examples**: See `tests/test.py` for comprehensive usage examples

---

*Made with ❤️ for the pandas community - Simplifying data type detection across cultures and locales*
