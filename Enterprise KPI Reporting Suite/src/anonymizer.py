"""
Data Anonymization Module for Enterprise KPI Reporting Suite
Handles PII anonymization to ensure data privacy compliance.

Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import hashlib
import pandas as pd
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAnonymizer:
    """
    Handles anonymization of sensitive data fields for privacy compliance.
    
    This class provides methods to hash or mask personally identifiable
    information (PII) such as Employee IDs before public sharing.
    
    Attributes:
        df (pd.DataFrame): The dataframe to anonymize
        anonymized_columns (List[str]): List of columns that have been anonymized
    """
    
    # Class-level constant for author metadata
    AUTHOR = "OGHENEOCHUKO EMMANUEL OGIDIAGBA"
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataAnonymizer with a dataframe.
        
        Args:
            df: The pandas DataFrame containing data to anonymize
        """
        self.df = df.copy()
        self.anonymized_columns: List[str] = []
        self._hash_salt = "enterprise_kpi_suite_2024"
    
    def hash_employee_id(self, column: str = 'Employee_ID', 
                         new_column_name: Optional[str] = None) -> pd.DataFrame:
        """
        Anonymize Employee_ID column using SHA-256 hashing.
        
        Args:
            column: Name of the column containing employee IDs
            new_column_name: Optional name for the anonymized column.
                           If None, replaces the original column.
        
        Returns:
            pd.DataFrame: DataFrame with anonymized employee IDs
        """
        if column not in self.df.columns:
            logger.warning(f"Column '{column}' not found in dataframe")
            return self.df
        
        def hash_value(val):
            """Hash a single value with salt."""
            if pd.isna(val):
                return val
            salted_value = f"{self._hash_salt}_{val}"
            return hashlib.sha256(salted_value.encode()).hexdigest()[:16]
        
        target_col = new_column_name if new_column_name else column
        
        self.df[target_col] = self.df[column].apply(hash_value)
        self.anonymized_columns.append(target_col)
        
        logger.info(f"Anonymized column '{column}' -> '{target_col}'")
        
        return self.df
    
    def mask_employee_id(self, column: str = 'Employee_ID',
                         show_chars: int = 3) -> pd.DataFrame:
        """
        Mask Employee_ID values, showing only first N characters.
        
        Args:
            column: Name of the column containing employee IDs
            show_chars: Number of characters to show at the start
        
        Returns:
            pd.DataFrame: DataFrame with masked employee IDs
        """
        if column not in self.df.columns:
            logger.warning(f"Column '{column}' not found in dataframe")
            return self.df
        
        def mask_value(val):
            """Mask a single value."""
            if pd.isna(val):
                return val
            val_str = str(val)
            if len(val_str) <= show_chars:
                return '*' * len(val_str)
            return val_str[:show_chars] + '*' * (len(val_str) - show_chars)
        
        self.df[column] = self.df[column].apply(mask_value)
        self.anonymized_columns.append(column)
        
        logger.info(f"Masked column '{column}' showing first {show_chars} chars")
        
        return self.df
    
    def anonymize_multiple_columns(self, columns: List[str],
                                   method: str = 'hash') -> pd.DataFrame:
        """
        Anonymize multiple columns using the specified method.
        
        Args:
            columns: List of column names to anonymize
            method: Either 'hash' or 'mask'
        
        Returns:
            pd.DataFrame: DataFrame with anonymized columns
        """
        for col in columns:
            if method == 'hash':
                self.hash_employee_id(col)
            elif method == 'mask':
                self.mask_employee_id(col)
        
        return self.df
    
    def get_anonymization_report(self) -> dict:
        """
        Generate a report of anonymization actions taken.
        
        Returns:
            dict: Report containing anonymized columns and metadata
        """
        return {
            'author': self.AUTHOR,
            'anonymized_columns': self.anonymized_columns,
            'total_columns_anonymized': len(self.anonymized_columns),
            'method_used': 'SHA-256 hash with salt' if 'Employee_ID' in self.anonymized_columns else 'None',
            'privacy_compliant': len(self.anonymized_columns) > 0
        }
    
    def get_anonymized_dataframe(self) -> pd.DataFrame:
        """
        Return the anonymized dataframe.
        
        Returns:
            pd.DataFrame: The anonymized dataframe
        """
        return self.df
