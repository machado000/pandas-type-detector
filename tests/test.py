"""
Comprehensive pytest test suite for pandas-type-detector package.

This module tests all type detection and conversion methods using realistic data
similar to the example CSV file to ensure proper functionality.
"""

import pytest
import pandas as pd
from pandas_type_detector import TypeDetectionPipeline, DataType


class TestBasicFunctionality:
    """Test basic pipeline creation and core functionality."""

    def test_pipeline_creation(self):
        """Test that pipelines can be created with different locales."""
        en_pipeline = TypeDetectionPipeline(locale="en-us")
        pt_pipeline = TypeDetectionPipeline(locale="pt-br")

        assert en_pipeline is not None
        assert pt_pipeline is not None
        assert en_pipeline.locale_config.name == "en-us"
        assert pt_pipeline.locale_config.name == "pt-br"


class TestNumericDetection:
    """Test numeric type detection and conversion."""

    @pytest.fixture
    def en_pipeline(self):
        return TypeDetectionPipeline(locale="en-us", on_error="coerce")

    @pytest.fixture
    def pt_pipeline(self):
        return TypeDetectionPipeline(locale="pt-br", on_error="coerce")

    def test_integer_detection_en_us(self, en_pipeline):
        """Test integer detection with en-us locale."""
        data = pd.Series(['1', '2', '3', '10', '100'], dtype=str)

        result = en_pipeline.detect_column_type(data)
        assert result.data_type == DataType.INTEGER
        assert result.confidence >= 0.8

    def test_float_detection_en_us(self, en_pipeline):
        """Test float detection with en-us locale."""
        data = pd.Series(['305.0', '300.5', '800.25', '45.75'], dtype=str)

        result = en_pipeline.detect_column_type(data)
        assert result.data_type == DataType.FLOAT
        assert result.confidence >= 0.8

    def test_float_detection_pt_br(self, pt_pipeline):
        """Test float detection with pt-br locale."""
        data = pd.Series(['305,50', '300,75', '800,25', '45,99'], dtype=str)

        result = pt_pipeline.detect_column_type(data)
        assert result.data_type == DataType.FLOAT
        assert result.confidence >= 0.8

    def test_text_with_numbers_rejection(self, en_pipeline):
        """Test that text containing numbers is properly rejected as numeric."""
        # This was the problematic data that was incorrectly detected before
        data = pd.Series([
            '(31) De 28/jul a 3/ago',
            '(31) De 28/jul a 3/ago',
            '(32) De 4/ago a 10/ago'
        ], dtype=str)

        result = en_pipeline.detect_column_type(data)
        # Should NOT be detected as numeric
        assert result.data_type != DataType.INTEGER
        assert result.data_type != DataType.FLOAT

    def test_numeric_conversion_en_us(self, en_pipeline):
        """Test numeric conversion with en-us locale."""
        # Test integer conversion
        int_data = pd.Series(['1', '2', '3', '10'], dtype=str)
        df_int = pd.DataFrame({'numbers': int_data})
        converted_int = en_pipeline.fix_dataframe_dtypes(df_int)
        assert converted_int['numbers'].dtype == 'Int64'

        # Test float conversion
        float_data = pd.Series(['1.5', '2.7', '3.14', '10.0'], dtype=str)
        df_float = pd.DataFrame({'decimals': float_data})
        converted_float = en_pipeline.fix_dataframe_dtypes(df_float)
        assert converted_float['decimals'].dtype == 'float64'

    def test_numeric_conversion_pt_br(self, pt_pipeline):
        """Test numeric conversion with pt-br locale."""
        data = pd.Series(['1,5', '2,7', '3,14', '10,0'], dtype=str)
        df = pd.DataFrame({'decimais': data})
        converted = pt_pipeline.fix_dataframe_dtypes(df)
        assert converted['decimais'].dtype == 'float64'
        assert abs(converted['decimais'].iloc[0] - 1.5) < 0.001


class TestBooleanDetection:
    """Test boolean type detection and conversion."""

    @pytest.fixture
    def en_pipeline(self):
        return TypeDetectionPipeline(locale="en-us", on_error="coerce")

    @pytest.fixture
    def pt_pipeline(self):
        return TypeDetectionPipeline(locale="pt-br", on_error="coerce")

    def test_boolean_detection_en_us(self, en_pipeline):
        """Test English boolean detection."""
        data = pd.Series(['true', 'false', 'yes', 'no'], dtype=str)

        result = en_pipeline.detect_column_type(data)
        assert result.data_type == DataType.BOOLEAN
        assert result.confidence >= 0.8

    def test_boolean_detection_pt_br(self, pt_pipeline):
        """Test Portuguese boolean detection."""
        data = pd.Series(['Sim', 'Não', 'sim', 'não'], dtype=str)

        result = pt_pipeline.detect_column_type(data)
        assert result.data_type == DataType.BOOLEAN
        assert result.confidence >= 0.8

    def test_boolean_conversion_en_us(self, en_pipeline):
        """Test English boolean conversion."""
        data = pd.Series(['true', 'false', 'yes', 'no'], dtype=str)
        df = pd.DataFrame({'flags': data})
        converted = en_pipeline.fix_dataframe_dtypes(df)

        assert converted['flags'].dtype == 'boolean'
        assert converted['flags'].iloc[0]  # Should be True
        assert not converted['flags'].iloc[1]  # Should be False

    def test_boolean_conversion_pt_br(self, pt_pipeline):
        """Test Portuguese boolean conversion."""
        data = pd.Series(['Sim', 'Não', 'sim', 'não'], dtype=str)
        df = pd.DataFrame({'considerar': data})
        converted = pt_pipeline.fix_dataframe_dtypes(df)

        assert converted['considerar'].dtype == 'boolean'
        assert converted['considerar'].iloc[0]  # 'Sim' -> True
        assert not converted['considerar'].iloc[1]  # 'Não' -> False


class TestDateTimeDetection:
    """Test datetime type detection and conversion."""

    @pytest.fixture
    def pipeline(self):
        return TypeDetectionPipeline(locale="en-us", on_error="coerce")

    def test_datetime_detection_iso_format(self, pipeline):
        """Test datetime detection with ISO format."""
        data = pd.Series(['2025-07-28', '2025-07-31', '2025-07-30'], dtype=str)

        result = pipeline.detect_column_type(data)
        assert result.data_type == DataType.DATETIME
        assert result.confidence >= 0.7

    def test_datetime_conversion(self, pipeline):
        """Test datetime conversion."""
        data = pd.Series(['2025-07-28', '2025-07-31', '2025-07-30'], dtype=str)
        df = pd.DataFrame({'dates': data})
        converted = pipeline.fix_dataframe_dtypes(df)

        assert pd.api.types.is_datetime64_any_dtype(converted['dates'])


class TestSkipColumns:
    """Test skip_columns functionality."""

    @pytest.fixture
    def pipeline(self):
        return TypeDetectionPipeline(locale="en-us", on_error="coerce")

    def test_skip_columns_functionality(self, pipeline):
        """Test that specified columns are skipped during conversion."""
        df = pd.DataFrame({
            'convert_this': ['1.5', '2.7', '3.14'],  # Should become float
            'skip_this': ['123', '456', '789'],      # Should remain object
            'also_skip': ['2025-01-01', '2025-01-02', '2025-01-03']  # Should remain object
        }, dtype=str)

        converted = pipeline.fix_dataframe_dtypes(
            df,
            skip_columns=['skip_this', 'also_skip']
        )

        # Converted column should change type
        assert converted['convert_this'].dtype == 'float64'

        # Skipped columns should remain as object
        assert converted['skip_this'].dtype == 'object'
        assert converted['also_skip'].dtype == 'object'


class TestRealisticDataScenarios:
    """Test with realistic data scenarios mimicking the example CSV."""

    @pytest.fixture
    def sample_csv_data(self):
        """Sample data similar to example_data.csv."""
        return pd.DataFrame({
            'atend': ['101832', '101841', '101855', '101867'],
            'celular': ['11999692813', '11999727372', '11987654321', '11976543210'],
            'email': ['user1@email.com', 'user2@email.com', 'user3@email.com', 'user4@email.com'],
            'cd_proced': ['32814166', '32301620', '39000025', '5738221'],
            'procedimento': ['ULTRASSONOGRAFIA', 'ECOCARDIOGRAMA', 'CATETER', 'CITRATO'],
            'dt_lanca': ['2025-07-28', '2025-07-31', '2025-07-30', '2025-07-29'],
            'hora': ['11:59:00', '12:03:00', '00:19:00', '09:00:00'],
            'qtd': ['1', '1', '2', '1'],
            'vl_uni': ['305.0', '300.0', '800.0', '45.0'],
            'vl_total': ['305.0', '300.0', '1600.0', '45.0'],
            'ajuste_unidade': ['1.0', '1.0', '1.0', '0.5'],
            'ajuste_semana': [
                '(31) De 28/jul a 3/ago', '(31) De 28/jul a 3/ago',
                '(32) De 4/ago a 10/ago', '(31) De 28/jul a 3/ago'
            ],
            'considerar_en': ['Yes', 'No', 'Yes', 'No'],
            'considerar_pt': ['Sim', 'Não', 'Sim', 'Não']
        }, dtype=str)

    def test_full_processing_en_us_locale(self, sample_csv_data):
        """Test complete processing with en-us locale."""
        pipeline = TypeDetectionPipeline(locale="en-us", on_error="coerce")

        # Skip text-heavy columns and the problematic ajuste_semana
        skip_cols = [
            'atend', 'celular', 'email', 'cd_proced', 'procedimento',
            'hora', 'ajuste_semana', 'considerar_pt'
        ]

        converted = pipeline.fix_dataframe_dtypes(sample_csv_data, skip_columns=skip_cols)

        # Check expected type conversions
        assert pd.api.types.is_datetime64_any_dtype(converted['dt_lanca'])
        assert converted['qtd'].dtype == 'Int64'
        # vl_uni has .0 decimals so may be detected as integer - that's acceptable
        assert converted['vl_uni'].dtype in ['float64', 'Int64']
        assert converted['vl_total'].dtype in ['float64', 'Int64']
        assert converted['ajuste_unidade'].dtype == 'float64'
        assert converted['considerar_en'].dtype == 'boolean'

        # Verify skipped columns remain as object
        assert converted['ajuste_semana'].dtype == 'object'
        assert converted['atend'].dtype == 'object'

    def test_full_processing_pt_br_locale(self, sample_csv_data):
        """Test complete processing with pt-br locale."""
        pipeline = TypeDetectionPipeline(locale="pt-br", on_error="coerce")

        # Skip text-heavy columns and English boolean
        skip_cols = [
            'atend', 'celular', 'email', 'cd_proced', 'procedimento',
            'hora', 'ajuste_semana', 'considerar_en'
        ]

        converted = pipeline.fix_dataframe_dtypes(sample_csv_data, skip_columns=skip_cols)

        # Check Portuguese boolean conversion
        assert converted['considerar_pt'].dtype == 'boolean'
        assert converted['considerar_pt'].iloc[0]  # 'Sim' -> True
        assert not converted['considerar_pt'].iloc[1]  # 'Não' -> False

    def test_problematic_ajuste_semana_column(self, sample_csv_data):
        """Test that the problematic ajuste_semana column is handled correctly."""
        pipeline = TypeDetectionPipeline(locale="en-us", on_error="coerce")

        # Test just the problematic column
        ajuste_data = sample_csv_data['ajuste_semana']
        result = pipeline.detect_column_type(ajuste_data)

        # Should NOT be detected as numeric
        assert result.data_type != DataType.INTEGER
        assert result.data_type != DataType.FLOAT

        # When processed in full dataframe, should remain as object
        df_single = pd.DataFrame({'ajuste_semana': ajuste_data})
        converted = pipeline.fix_dataframe_dtypes(df_single)
        assert converted['ajuste_semana'].dtype == 'object'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
