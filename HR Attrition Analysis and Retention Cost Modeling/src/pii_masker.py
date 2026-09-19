"""
PII Masking Module

Provides data privacy compliance by anonymizing sensitive fields
before public sharing or analysis.
"""

import pandas as pd
import hashlib
from typing import List, Optional
from datetime import datetime


class PIIMasker:
    """
    Handles anonymization of Personally Identifiable Information (PII)
    in HR datasets to ensure data privacy compliance.
    
    Supports multiple masking strategies including hashing, redaction,
    and pseudonymization.
    """
    
    # Default PII columns in HR datasets
    DEFAULT_PII_COLUMNS = [
        'EmployeeNumber',
        'EmployeeCount'
    ]
    
    def __init__(self, salt: Optional[str] = None):
        """
        Initialize the PIIMasker.
        
        Args:
            salt: Optional salt value for hashing. If not provided,
                  uses current timestamp for uniqueness.
        """
        self.salt = salt or str(datetime.now().timestamp())
        self.masked_columns: List[str] = []
    
    def mask(self, df: pd.DataFrame, 
             columns: Optional[List[str]] = None,
             strategy: str = 'hash') -> pd.DataFrame:
        """
        Apply PII masking to specified columns.
        
        Args:
            df: The DataFrame containing PII data.
            columns: List of column names to mask. If None, uses defaults.
            strategy: Masking strategy ('hash', 'redact', 'pseudonymize').
            
        Returns:
            pd.DataFrame: DataFrame with masked columns.
        """
        if columns is None:
            columns = self.DEFAULT_PII_COLUMNS
        
        # Filter to only columns that exist in the DataFrame
        columns_to_mask = [col for col in columns if col in df.columns]
        
        df_masked = df.copy()
        
        for col in columns_to_mask:
            if strategy == 'hash':
                df_masked[col] = self._hash_column(df_masked[col])
            elif strategy == 'redact':
                df_masked[col] = self._redact_column(df_masked[col])
            elif strategy == 'pseudonymize':
                df_masked[col] = self._pseudonymize_column(df_masked[col])
            else:
                raise ValueError(f"Unknown masking strategy: {strategy}")
            
            self.masked_columns.append(col)
        
        return df_masked
    
    def _hash_column(self, series: pd.Series) -> pd.Series:
        """
        Apply SHA-256 hashing to a column.
        
        Args:
            series: The pandas Series to hash.
            
        Returns:
            pd.Series: Hashed values with prefix 'EMP_'.
        """
        def hash_value(val):
            if pd.isna(val):
                return 'UNKNOWN'
            combined = f"{self.salt}_{val}"
            hash_obj = hashlib.sha256(combined.encode())
            return f"EMP_{hash_obj.hexdigest()[:12]}"
        
        return series.apply(hash_value)
    
    def _redact_column(self, series: pd.Series) -> pd.Series:
        """
        Redact all values in a column.
        
        Args:
            series: The pandas Series to redact.
            
        Returns:
            pd.Series: Redacted values.
        """
        return pd.Series(['[REDACTED]' for _ in range(len(series))], index=series.index)
    
    def _pseudonymize_column(self, series: pd.Series) -> pd.Series:
        """
        Replace values with sequential pseudonyms.
        
        Args:
            series: The pandas Series to pseudonymize.
            
        Returns:
            pd.Series: Pseudonymized values.
        """
        unique_values = series.unique()
        mapping = {val: f'ID_{i+1:04d}' for i, val in enumerate(unique_values)}
        
        return series.map(mapping)
    
    def get_masking_report(self) -> dict:
        """
        Generate a report of applied masking operations.
        
        Returns:
            dict: Report containing masked columns and strategy used.
        """
        return {
            'masked_columns': self.masked_columns,
            'masking_strategy': 'hash',
            'salt_used': self.salt[:8] + '...' if len(self.salt) > 8 else self.salt,
            'timestamp': datetime.now().isoformat()
        }
    
    def verify_masking(self, df: pd.DataFrame, 
                       original_values: Optional[List] = None) -> bool:
        """
        Verify that masking has been properly applied.
        
        Args:
            df: The DataFrame to verify.
            original_values: Optional list of original values to check against.
            
        Returns:
            bool: True if masking is verified, False otherwise.
        """
        for col in self.masked_columns:
            if col not in df.columns:
                return False
            
            # Check that no original numeric employee numbers remain
            if original_values:
                for orig_val in original_values:
                    if orig_val in df[col].values:
                        return False
            
            # Check that values are properly formatted (start with EMP_)
            sample_value = df[col].iloc[0] if len(df) > 0 else None
            if sample_value and not str(sample_value).startswith('EMP_'):
                return False
        
        return True
    
    def create_masking_audit_log(self, df_original: pd.DataFrame,
                                  df_masked: pd.DataFrame) -> dict:
        """
        Create an audit log for compliance purposes.
        
        Args:
            df_original: Original DataFrame before masking.
            df_masked: DataFrame after masking.
            
        Returns:
            dict: Audit log with transformation details.
        """
        audit_log = {
            'audit_timestamp': datetime.now().isoformat(),
            'original_row_count': len(df_original),
            'masked_row_count': len(df_masked),
            'columns_masked': self.masked_columns,
            'transformations': {}
        }
        
        for col in self.masked_columns:
            if col in df_original.columns and col in df_masked.columns:
                audit_log['transformations'][col] = {
                    'unique_original_values': df_original[col].nunique(),
                    'unique_masked_values': df_masked[col].nunique(),
                    'sample_original': str(df_original[col].head(3).tolist()),
                    'sample_masked': str(df_masked[col].head(3).tolist())
                }
        
        return audit_log
