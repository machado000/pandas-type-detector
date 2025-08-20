# Pandas Type Detector

A modular, extensible system for automatically detecting and converting pandas DataFrame column types using a strategy pattern with confidence scoring and locale-aware parsing.

## Features

- **Locale-aware parsing**: Built-in support for PT-BR and EN-US numeric formats
- **Modular architecture**: Each detector handles a specific data type
- **Confidence scoring**: Get confidence levels for type detection decisions
- **Error handling modes**: Choose how to handle conversion errors (`coerce`, `raise`, `ignore`)
- **Excel compatibility**: Correctly handles data that Excel might misinterpret
- **Extensible design**: Easy to add new locales or data types
- **Production ready**: Successfully tested with 14,607+ rows in production

## Installation

```bash
pip install pandas-type-detector
```

## Quick Start

```python
from pandas_type_detector import TypeDetectionPipeline
import pandas as pd

# Create sample DataFrame with PT-BR numeric data
data = {
    'receita': ['1.364,00', '343', '111,1', '1.950,00'],
    'nome': ['João', 'Maria', 'Pedro', 'Ana'],
    'ativo': ['sim', 'não', 'sim', 'não'],
    'data': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
}

df = pd.DataFrame(data)

# Initialize the pipeline with PT-BR locale
pipeline = TypeDetectionPipeline(locale="pt-br")

# Automatically detect and convert all column types
df_converted = pipeline.fix_dataframe_dtypes(df)

print(df_converted.dtypes)
# Output:
# receita      float64
# nome          string  
# ativo       boolean
# data     datetime64[ns]
```

## Locale Support

### PT-BR (Portuguese Brazil)
- **Decimal separator**: `,` (comma)
- **Thousands separator**: `.` (dot)
- **Currency**: `R$`, `BRL`
- **Boolean values**: `sim`/`não`, `verdadeiro`/`falso`
- **Date formats**: `DD/MM/YYYY`, `YYYY-MM-DD`

### EN-US (English United States)
- **Decimal separator**: `.` (dot)
- **Thousands separator**: `,` (comma)
- **Currency**: `$`, `USD`
- **Boolean values**: `yes`/`no`, `true`/`false`
- **Date formats**: `MM/DD/YYYY`, `YYYY-MM-DD`

## Advanced Usage

### Error Handling Modes

```python
# Coerce errors (default) - convert invalid values to NaN
pipeline = TypeDetectionPipeline(locale="pt-br", on_error="coerce")

# Raise exceptions on conversion errors
pipeline = TypeDetectionPipeline(locale="pt-br", on_error="raise")

# Ignore problematic columns - leave them unchanged
pipeline = TypeDetectionPipeline(locale="pt-br", on_error="ignore")
```

### Individual Column Detection

```python
# Get detailed information about type detection
result = pipeline.detect_column_type(df['receita'])

print(f"Detected type: {result.data_type.value}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Metadata: {result.metadata}")
```

### Skip Specific Columns

```python
# Skip certain columns during conversion
df_converted = pipeline.fix_dataframe_dtypes(
    df, 
    skip_columns=['keep_as_string', 'manual_column']
)
```

## Data Type Detection

The package detects the following data types:

| Data Type | Description | Examples |
|-----------|-------------|----------|
| `INTEGER` | Whole numbers | `123`, `1.000` (PT-BR) |
| `FLOAT` | Decimal numbers | `123,45` (PT-BR), `123.45` (EN-US) |
| `BOOLEAN` | True/false values | `sim`/`não`, `yes`/`no` |
| `DATETIME` | Date and time | `2024-01-01`, `01/02/2024` |
| `TEXT` | String data | Any non-matching text |

## Real-World Examples

### Excel Import Fix

```python
# Excel often misinterprets PT-BR numbers as dates
# This package correctly identifies and converts them

excel_data = ['1.364,00', '2.500,75', '3.100,25']
df = pd.DataFrame({'revenue': excel_data})

pipeline = TypeDetectionPipeline(locale="pt-br")
df_fixed = pipeline.fix_dataframe_dtypes(df)

# Now revenue is properly converted to float64
print(df_fixed['revenue'].tolist())
# Output: [1364.0, 2500.75, 3100.25]
```

### Mixed Data Handling

```python
# Handle mixed valid/invalid data gracefully
messy_data = ['1.364,00', 'invalid', '111,1', '-', '', '1.950,00']
df = pd.DataFrame({'values': messy_data})

pipeline = TypeDetectionPipeline(locale="pt-br", on_error="coerce")
df_clean = pipeline.fix_dataframe_dtypes(df)

# Invalid values become NaN, valid ones are converted
print(df_clean['values'].tolist())
# Output: [1364.0, NaN, 111.1, NaN, NaN, 1950.0]
```

### Production ETL Pipeline

```python
def process_financial_data(df):
    """Production ETL function using type detector."""
    
    # Configure for strict error handling in production
    pipeline = TypeDetectionPipeline(
        locale="pt-br", 
        on_error="raise",
        sample_size=1000
    )
    
    try:
        # Convert all columns automatically
        df_processed = pipeline.fix_dataframe_dtypes(df)
        
        # Log conversion results
        print(f"Successfully processed {len(df_processed)} rows")
        return df_processed
        
    except Exception as e:
        print(f"Data quality issue detected: {e}")
        raise
```

## Architecture

The package uses a modular strategy pattern:

```python
# Each detector handles one specific data type
from pandas_type_detector import (
    NumericDetector,    # Handles PT-BR/EN-US numbers
    BooleanDetector,    # Handles locale-specific booleans  
    DateTimeDetector,   # Handles various date formats
    TextDetector        # Fallback for text data
)

# All coordinated by the main pipeline
pipeline = TypeDetectionPipeline(locale="pt-br")
```

## Configuration

### Custom Sample Size

```python
# Use larger sample for better accuracy on big datasets
pipeline = TypeDetectionPipeline(
    locale="pt-br",
    sample_size=5000  # Default: 1000
)
```

### Confidence Thresholds

```python
# Access individual detectors for custom configuration
from pandas_type_detector import NumericDetector, LOCALES

detector = NumericDetector(
    locale_config=LOCALES["pt-br"],
    min_confidence=0.9  # Higher threshold
)
```

## Testing

Run the comprehensive test suite:

```bash
cd pandas-type-detector
python tests/test.py
```

The test suite includes:
- PT-BR numeric format validation
- Error handling verification
- Boolean detection tests
- DateTime parsing tests
- Excel compatibility tests
- Real-world scenario validation

## Performance

- **Optimized sampling**: Only analyzes a configurable sample of rows
- **Early exit**: Stops detection when high confidence is reached
- **Minimal overhead**: Designed for production ETL pipelines
- **Memory efficient**: Processes columns independently

## Contributing

The modular design makes it easy to contribute:

### Adding a New Locale

```python
from pandas_type_detector import LOCALES, LocaleConfig

# Add Spanish locale
LOCALES['es-es'] = LocaleConfig(
    name='es-es',
    decimal_separator=',',
    thousands_separator='.',
    currency_symbols=['€', 'EUR'],
    date_formats=[r'^\d{1,2}/\d{1,2}/\d{4}$']
)
```

### Adding a New Detector

```python
from pandas_type_detector import TypeDetector, DataType, DetectionResult

class URLDetector(TypeDetector):
    def detect(self, series):
        # Implementation here
        pass
    
    def convert(self, series):
        # Implementation here  
        pass
```

## Requirements

- Python 3.7+
- pandas >= 1.0.0
- numpy >= 1.19.0

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Developed to solve real-world data quality issues in Brazilian financial data processing. Successfully deployed in production handling 14,607+ rows of complex PT-BR formatted data.

## Support

- **Issues**: GitHub Issues
- **Documentation**: This README
- **Examples**: See `tests/test.py` for comprehensive usage examples

---

*Made with ❤️ for the pandas community*
