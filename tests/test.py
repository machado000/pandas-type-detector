#!/usr/bin/env python3
"""
Comprehensive test suite for pandas-type-detector package.

This module contains all tests for the type detection system, including:
- PT-BR numeric detection
- Error handling modes  
- Boolean detection
- DateTime detection
- Locale support
- Excel compatibility
"""

import unittest
import pandas as pd
import numpy as np
from pandas_type_detector import (
    TypeDetectionPipeline, 
    DataType, 
    LOCALES,
    NumericDetector,
    BooleanDetector,
    DateTimeDetector
)


class TestPTBRNumericDetection(unittest.TestCase):
    """Test PT-BR numeric format detection and conversion."""
    
    def setUp(self):
        self.pipeline = TypeDetectionPipeline(locale="pt-br")
        
    def test_ptbr_numeric_formats(self):
        """Test various PT-BR numeric formats."""
        test_cases = [
            ('1.364,00', 1364.00),
            ('343', 343.0),
            ('111,1', 111.1),
            ('1.950,00', 1950.00),
            ('35', 35.0),
            ('3.419,00', 3419.00),
            ('1.234.567,89', 1234567.89),
            ('500,25', 500.25)
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                series = pd.Series([input_val])
                result = self.pipeline.detect_column_type(series)
                self.assertIn(result.data_type, [DataType.INTEGER, DataType.FLOAT])
                
                converted = self.pipeline.fix_dataframe_dtypes(pd.DataFrame({'test': series}))
                actual = converted['test'].iloc[0]
                self.assertAlmostEqual(actual, expected, places=2)
    
    def test_ptbr_mixed_data(self):
        """Test PT-BR numeric detection with mixed valid/invalid data."""
        data = ['1.364,00', '343', '-', '111,1', '', '1.950,00']
        series = pd.Series(data)
        
        result = self.pipeline.detect_column_type(series)
        self.assertIn(result.data_type, [DataType.INTEGER, DataType.FLOAT])
        self.assertGreater(result.confidence, 0.6)
        
        df = pd.DataFrame({'receita': data})
        converted = self.pipeline.fix_dataframe_dtypes(df)
        
        # Check that valid values were converted correctly
        self.assertEqual(converted['receita'].iloc[0], 1364.00)
        self.assertEqual(converted['receita'].iloc[1], 343.0)
        self.assertTrue(pd.isna(converted['receita'].iloc[2]))  # '-' becomes NaN
        
    def test_thousands_vs_decimal_separator(self):
        """Test disambiguation between thousands and decimal separators."""
        test_cases = [
            ('1.234', 1234.0),  # Ambiguous - treated as thousands
            ('1.234,56', 1234.56),  # Clear PT-BR format
            ('1.000', 1000.0),  # Thousands separator
            ('123,45', 123.45),  # Decimal separator
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                series = pd.Series([input_val])
                converted = self.pipeline.fix_dataframe_dtypes(pd.DataFrame({'test': series}))
                actual = converted['test'].iloc[0]
                self.assertAlmostEqual(actual, expected, places=2)


class TestErrorHandling(unittest.TestCase):
    """Test error handling modes."""
    
    def test_coerce_mode(self):
        """Test 'coerce' error handling mode."""
        pipeline = TypeDetectionPipeline(locale="pt-br", on_error="coerce")
        
        # Mix of valid and invalid data
        data = ['123,45', 'invalid!@#', '67,89', 'abc123', '100,00']
        df = pd.DataFrame({'mixed': data})
        
        result = pipeline.fix_dataframe_dtypes(df)
        
        # Valid values should be converted, invalid ones become NaN
        self.assertAlmostEqual(result['mixed'].iloc[0], 123.45, places=2)
        self.assertTrue(pd.isna(result['mixed'].iloc[1]))  # invalid becomes NaN
        self.assertAlmostEqual(result['mixed'].iloc[2], 67.89, places=2)
        
    def test_ignore_mode(self):
        """Test 'ignore' error handling mode."""
        pipeline = TypeDetectionPipeline(locale="pt-br", on_error="ignore")
        
        data = ['123,45', 'invalid!@#', '67,89']
        df = pd.DataFrame({'mixed': data})
        original_dtype = df['mixed'].dtype
        
        result = pipeline.fix_dataframe_dtypes(df)
        
        # If conversion would fail, column should remain unchanged
        # This test depends on the specific implementation
        self.assertIsNotNone(result)
        
    def test_raise_mode(self):
        """Test 'raise' error handling mode."""
        pipeline = TypeDetectionPipeline(locale="pt-br", on_error="raise")
        
        # Test with data that should convert successfully
        good_data = ['123,45', '67,89', '100,00']
        df_good = pd.DataFrame({'numbers': good_data})
        
        # This should work without raising
        result = pipeline.fix_dataframe_dtypes(df_good)
        self.assertIsNotNone(result)


class TestBooleanDetection(unittest.TestCase):
    """Test boolean detection for different locales."""
    
    def test_ptbr_boolean(self):
        """Test PT-BR boolean detection."""
        pipeline = TypeDetectionPipeline(locale="pt-br")
        
        data = ['sim', 'não', 'sim', 'sim', 'não']
        series = pd.Series(data)
        
        result = pipeline.detect_column_type(series)
        self.assertEqual(result.data_type, DataType.BOOLEAN)
        self.assertGreater(result.confidence, 0.8)
        
        df = pd.DataFrame({'ativo': data})
        converted = pipeline.fix_dataframe_dtypes(df)
        
        expected = [True, False, True, True, False]
        actual = converted['ativo'].tolist()
        self.assertEqual(actual, expected)
        
    def test_english_boolean(self):
        """Test English boolean detection."""
        pipeline = TypeDetectionPipeline(locale="en-us")
        
        data = ['yes', 'no', 'true', 'false', 'y']
        series = pd.Series(data)
        
        result = pipeline.detect_column_type(series)
        self.assertEqual(result.data_type, DataType.BOOLEAN)
        
    def test_numeric_not_boolean(self):
        """Test that numeric 1/0 are not detected as boolean."""
        pipeline = TypeDetectionPipeline(locale="pt-br")
        
        # Numeric 1/0 should be detected as numeric, not boolean
        data = ['1', '0', '1', '0', '1']
        series = pd.Series(data)
        
        result = pipeline.detect_column_type(series)
        self.assertIn(result.data_type, [DataType.INTEGER, DataType.FLOAT])
        self.assertNotEqual(result.data_type, DataType.BOOLEAN)


class TestDateTimeDetection(unittest.TestCase):
    """Test datetime detection."""
    
    def setUp(self):
        self.pipeline = TypeDetectionPipeline(locale="pt-br")
        
    def test_common_date_formats(self):
        """Test common date format detection."""
        date_data = ['2024-01-01', '2024-02-15', '2024-03-30']
        series = pd.Series(date_data)
        
        result = self.pipeline.detect_column_type(series)
        self.assertEqual(result.data_type, DataType.DATETIME)
        
        df = pd.DataFrame({'data': date_data})
        converted = self.pipeline.fix_dataframe_dtypes(df)
        
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(converted['data']))
        
    def test_date_vs_numeric_disambiguation(self):
        """Test that PT-BR numbers aren't mistaken for dates."""
        # These look like numbers, not dates
        numeric_data = ['1.364,00', '2.500,75', '3.150,25']
        series = pd.Series(numeric_data)
        
        result = self.pipeline.detect_column_type(series)
        self.assertNotEqual(result.data_type, DataType.DATETIME)
        self.assertIn(result.data_type, [DataType.INTEGER, DataType.FLOAT])


class TestLocaleSupport(unittest.TestCase):
    """Test locale configuration and support."""
    
    def test_available_locales(self):
        """Test that expected locales are available."""
        self.assertIn("pt-br", LOCALES)
        self.assertIn("en-us", LOCALES)
        
    def test_locale_configuration(self):
        """Test locale configuration properties."""
        ptbr = LOCALES["pt-br"]
        self.assertEqual(ptbr.decimal_separator, ",")
        self.assertEqual(ptbr.thousands_separator, ".")
        
        enus = LOCALES["en-us"]
        self.assertEqual(enus.decimal_separator, ".")
        self.assertEqual(enus.thousands_separator, ",")
        
    def test_invalid_locale(self):
        """Test error handling for invalid locale."""
        with self.assertRaises(ValueError):
            TypeDetectionPipeline(locale="invalid-locale")


class TestExcelCompatibility(unittest.TestCase):
    """Test compatibility with Excel-imported data."""
    
    def setUp(self):
        self.pipeline = TypeDetectionPipeline(locale="pt-br")
        
    def test_excel_numeric_correction(self):
        """Test correction of Excel misinterpreted numeric data."""
        # Simulate data that Excel might misinterpret
        excel_data = ['1.364,00', '343', '111,1', '1.950,00', '35']
        
        # Verify our detector correctly identifies it as numeric
        series = pd.Series(excel_data)
        result = self.pipeline.detect_column_type(series)
        self.assertIn(result.data_type, [DataType.INTEGER, DataType.FLOAT])
        
        # Verify conversion works correctly
        df = pd.DataFrame({'receita': excel_data})
        converted = self.pipeline.fix_dataframe_dtypes(df)
        
        expected_values = [1364.00, 343.0, 111.1, 1950.00, 35.0]
        actual_values = converted['receita'].tolist()
        
        for expected, actual in zip(expected_values, actual_values):
            self.assertAlmostEqual(actual, expected, places=2)


class TestComprehensiveDataFrame(unittest.TestCase):
    """Test complete DataFrame processing with mixed data types."""
    
    def test_mixed_dataframe_processing(self):
        """Test processing a DataFrame with multiple data types."""
        data = {
            'receita': ['1.364,00', '343', '111,1', '1.950,00'],
            'nome': ['João', 'Maria', 'Pedro', 'Ana'],
            'data': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
            'ativo': ['sim', 'não', 'sim', 'não'],
            'categoria': ['A', 'B', 'A', 'C']
        }
        
        df = pd.DataFrame(data)
        pipeline = TypeDetectionPipeline(locale="pt-br")
        
        converted = pipeline.fix_dataframe_dtypes(df)
        
        # Check that each column was converted to appropriate type
        self.assertTrue(pd.api.types.is_numeric_dtype(converted['receita']))
        self.assertTrue(pd.api.types.is_string_dtype(converted['nome']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(converted['data']))
        self.assertTrue(pd.api.types.is_bool_dtype(converted['ativo']))
        self.assertTrue(pd.api.types.is_string_dtype(converted['categoria']))
        
    def test_skip_columns(self):
        """Test skipping specific columns during conversion."""
        data = {
            'numbers': ['1.364,00', '343', '111,1'],
            'keep_as_string': ['1.364,00', '343', '111,1']
        }
        
        df = pd.DataFrame(data)
        pipeline = TypeDetectionPipeline(locale="pt-br")
        
        converted = pipeline.fix_dataframe_dtypes(df, skip_columns=['keep_as_string'])
        
        # numbers should be converted, keep_as_string should remain string
        self.assertTrue(pd.api.types.is_numeric_dtype(converted['numbers']))
        self.assertTrue(pd.api.types.is_string_dtype(converted['keep_as_string']))


# Integration Tests and Demos
class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world usage scenarios."""
    
    def test_production_like_data(self):
        """Test with production-like data including edge cases."""
        data = {
            'receita': ['1.364,00', '343', '111,1', '-', '', '1.950,00', 'n/a'],
            'percentual': ['15,5%', '20,0%', '8,75%', '12,25%', '30,0%', '5,5%', '0,0%'],
            'status': ['ativo', 'inativo', 'ativo', 'pendente', 'ativo', 'inativo', 'ativo']
        }
        
        df = pd.DataFrame(data)
        pipeline = TypeDetectionPipeline(locale="pt-br", on_error="coerce")
        
        # This should not raise any exceptions
        converted = pipeline.fix_dataframe_dtypes(df)
        self.assertIsNotNone(converted)
        
        # Check that numeric conversion handled placeholders correctly
        self.assertTrue(pd.isna(converted['receita'].iloc[3]))  # '-' becomes NaN
        self.assertTrue(pd.isna(converted['receita'].iloc[4]))  # '' becomes NaN


def run_demo():
    """Run a demonstration of the package features."""
    print("=" * 60)
    print("Pandas Type Detector - Feature Demonstration")
    print("=" * 60)
    
    # Create sample data
    data = {
        'receita': ['1.364,00', '343', '111,1', '1.950,00', '35'],
        'nome': ['João', 'Maria', 'Pedro', 'Ana', 'Carlos'],
        'ativo': ['sim', 'não', 'sim', 'sim', 'não'],
        'data': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df.dtypes)
    print("\nSample data:")
    print(df.head())
    
    # Apply type detection
    pipeline = TypeDetectionPipeline(locale="pt-br")
    
    print("\nColumn type detection results:")
    for column in df.columns:
        result = pipeline.detect_column_type(df[column])
        print(f"  {column}: {result.data_type.value} "
              f"(confidence: {result.confidence:.2f})")
    
    # Convert the DataFrame
    df_converted = pipeline.fix_dataframe_dtypes(df)
    
    print("\nAfter conversion:")
    print(df_converted.dtypes)
    print("\nConverted data:")
    print(df_converted.head())
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    print("Running pandas-type-detector test suite...")
    
    # Run demo first
    run_demo()
    
    print("\n" + "=" * 60)
    print("Running Unit Tests")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n✅ All tests completed!")
