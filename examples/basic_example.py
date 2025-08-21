"""
# Example script to demonstrate the use of pandas_type_detector for data cleaning and transformation
"""

import logging
import pandas as pd

from pandas_type_detector import TypeDetectionPipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def main():
    # Read CSV with all columns as strings to bypass pandas inference
    df = pd.read_csv('./examples/example_data.csv', index_col=False, dtype=str)

    # Use English locale for numeric columns with '.' decimal separator
    pipeline = TypeDetectionPipeline(locale="en-us", on_error="coerce")

    # Skip columns that should remain as text/object (IDs, codes, etc.)
    skip_columns = [
        'atend', 'celular', 'cd_proced', 'hora', 'lead_telefone', 'lead_email'
    ]

    # Let the algorithm detect and convert the remaining columns
    df = pipeline.fix_dataframe_dtypes(df, skip_columns=skip_columns)

    print("=== Main DataFrame Types (en-us locale) ===")
    print(df.dtypes)

    # Test the 'considerar' column separately with pt-br locale
    print("\n=== Testing 'considerar' column with pt-br locale ===")
    considerar_df = df[['considerar']].copy()
    pt_pipeline = TypeDetectionPipeline(locale="pt-br", on_error="coerce")
    considerar_df = pt_pipeline.fix_dataframe_dtypes(considerar_df, skip_columns=[])

    print("considerar column detection result:")
    print(f"dtype: {considerar_df['considerar'].dtype}")
    print(f"sample values: {considerar_df['considerar'].head().tolist()}")
    print(f"unique values: {considerar_df['considerar'].unique()}")


if __name__ == "__main__":
    main()
