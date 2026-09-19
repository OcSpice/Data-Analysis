"""
Data Loader Module
==================

Handles loading and initial parsing of financial transaction data.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class DataLoader:
    """
    Responsible for loading transaction data from CSV files.
    
    Attributes:
        file_path (Path): Path to the source data file.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the DataLoader with a file path.
        
        Args:
            file_path: Path to the CSV file containing transaction data.
        """
        self.file_path = Path(file_path)
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Loaded transaction data.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is empty or has no columns.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        # Load CSV - only parse dates if Date column exists
        df = pd.read_csv(
            self.file_path,
            dtype={
                "Transaction_ID": "string",
                "Year": "Int64",
                "Quarter": "string",
                "Department": "string",
                "Category": "string",
                "Transaction_Type": "string",
                "Source_System": "string",
                "Region": "string",
                "Currency": "string",
                "Status": "string"
            }
        )
        
        # Parse dates only if Date column exists
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        
        # Validate that data was loaded
        if df.empty:
            raise ValueError("Loaded data is empty")
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate a summary of the loaded data.
        
        Args:
            df: DataFrame to summarize.
            
        Returns:
            dict: Summary statistics including row count, column count,
                  date range, and memory usage.
        """
        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "date_range": {
                "min": df["Date"].min() if "Date" in df.columns else None,
                "max": df["Date"].max() if "Date" in df.columns else None
            },
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
