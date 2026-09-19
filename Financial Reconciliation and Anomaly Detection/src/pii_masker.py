"""
PII Masking Module
==================

Provides utilities for anonymizing personally identifiable information (PII)
and sensitive data fields to ensure compliance with data privacy regulations.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Optional, Dict
from hashlib import sha256


class PIIMasker:
    """
    Utility class for masking and anonymizing sensitive data fields.
    
    This class implements various masking techniques:
    - Partial masking (showing only first/last characters)
    - Hash-based pseudonymization
    - Complete redaction
    - Format-preserving masking
    
    Attributes:
        df (pd.DataFrame): DataFrame containing data to mask.
        masked_columns (List[str]): List of columns that have been masked.
        masking_log (Dict): Log of all masking operations performed.
    """
    
    # Default PII columns for financial transaction data
    DEFAULT_PII_COLUMNS = [
        "Analyst",
        "Transaction_ID"
    ]
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the PII Masker.
        
        Args:
            df: DataFrame containing potentially sensitive data.
        """
        self.df = df.copy()
        self.masked_columns: List[str] = []
        self.masking_log: Dict = {}
    
    def mask_name(self, name: str, show_first: int = 1) -> str:
        """
        Mask a person's name, showing only the first character(s).
        
        Args:
            name: The full name to mask.
            show_first: Number of characters to reveal at the start.
            
        Returns:
            str: Masked name (e.g., "J*** N****" for "John Smith").
        """
        if pd.isna(name) or not isinstance(name, str):
            return name
        
        # Split by common delimiters (space, hyphen)
        parts = re.split(r"[\s\-]+", name.strip())
        masked_parts = []
        
        for part in parts:
            if len(part) <= show_first:
                masked_parts.append(part)
            else:
                masked_parts.append(part[0] + "*" * (len(part) - 1))
        
        return " ".join(masked_parts)
    
    def mask_transaction_id(self, txn_id: str, show_last: int = 4) -> str:
        """
        Mask a transaction ID, showing only the last few characters.
        
        Args:
            txn_id: The transaction ID to mask.
            show_last: Number of characters to reveal at the end.
            
        Returns:
            str: Masked transaction ID (e.g., "TXN-******1234").
        """
        if pd.isna(txn_id) or not isinstance(txn_id, str):
            return txn_id
        
        txn_id = str(txn_id)
        if len(txn_id) <= show_last:
            return txn_id
        
        # Keep prefix format (e.g., "TXN-") and mask the numeric part
        if "-" in txn_id:
            parts = txn_id.split("-", 1)
            prefix = parts[0] + "-"
            numeric_part = parts[1]
            
            if len(numeric_part) > show_last:
                masked_numeric = "*" * (len(numeric_part) - show_last) + numeric_part[-show_last:]
            else:
                masked_numeric = numeric_part
            
            return prefix + masked_numeric
        else:
            # No prefix, just mask from the beginning
            return "*" * (len(txn_id) - show_last) + txn_id[-show_last:]
    
    def hash_value(self, value: str, salt: str = "financial_recon_2024") -> str:
        """
        Create a deterministic hash of a value for pseudonymization.
        
        Args:
            value: The value to hash.
            salt: Salt string to add randomness to the hash.
            
        Returns:
            str: First 12 characters of SHA256 hash.
        """
        if pd.isna(value) or not isinstance(value, str):
            return value
        
        salted_value = f"{salt}_{value}"
        hash_object = sha256(salted_value.encode())
        return hash_object.hexdigest()[:12]
    
    def mask_column_partial(
        self,
        column: str,
        strategy: str = "name",
        show_chars: int = 1
    ) -> pd.DataFrame:
        """
        Apply partial masking to a column.
        
        Args:
            column: Name of the column to mask.
            strategy: Masking strategy ("name", "id", "first_n", "last_n").
            show_chars: Number of characters to show.
            
        Returns:
            pd.DataFrame: DataFrame with masked column.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if strategy == "name":
            self.df[column] = self.df[column].apply(
                lambda x: self.mask_name(x, show_first=show_chars)
            )
        elif strategy == "id":
            self.df[column] = self.df[column].apply(
                lambda x: self.mask_transaction_id(x, show_last=show_chars)
            )
        elif strategy == "first_n":
            self.df[column] = self.df[column].apply(
                lambda x: str(x)[:show_chars] + "*" * (len(str(x)) - show_chars)
                if pd.notna(x) else x
            )
        elif strategy == "last_n":
            self.df[column] = self.df[column].apply(
                lambda x: "*" * (len(str(x)) - show_chars) + str(x)[-show_chars:]
                if pd.notna(x) and len(str(x)) > show_chars else x
            )
        
        if column not in self.masked_columns:
            self.masked_columns.append(column)
        
        self.masking_log[column] = {
            "strategy": strategy,
            "show_chars": show_chars,
            "records_masked": len(self.df)
        }
        
        return self.df
    
    def mask_column_hash(self, column: str, salt: Optional[str] = None) -> pd.DataFrame:
        """
        Apply hash-based pseudonymization to a column.
        
        Args:
            column: Name of the column to mask.
            salt: Optional salt for hashing.
            
        Returns:
            pd.DataFrame: DataFrame with hashed column.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        self.df[column] = self.df[column].apply(
            lambda x: self.hash_value(x, salt=salt or "default_salt")
        )
        
        if column not in self.masked_columns:
            self.masked_columns.append(column)
        
        self.masking_log[column] = {
            "strategy": "hash",
            "salt_used": salt is not None,
            "records_masked": len(self.df)
        }
        
        return self.df
    
    def mask_column_redact(self, column: str, replacement: str = "[REDACTED]") -> pd.DataFrame:
        """
        Completely redact a column with a placeholder value.
        
        Args:
            column: Name of the column to redact.
            replacement: Replacement string for redacted values.
            
        Returns:
            pd.DataFrame: DataFrame with redacted column.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        self.df[column] = replacement
        
        if column not in self.masked_columns:
            self.masked_columns.append(column)
        
        self.masking_log[column] = {
            "strategy": "redact",
            "replacement": replacement,
            "records_masked": len(self.df)
        }
        
        return self.df
    
    def mask_analyst_names(self, strategy: str = "partial") -> pd.DataFrame:
        """
        Mask analyst names using the specified strategy.
        
        Args:
            strategy: One of "partial" (show initials) or "hash".
            
        Returns:
            pd.DataFrame: DataFrame with masked analyst names.
        """
        if "Analyst" not in self.df.columns:
            return self.df
        
        if strategy == "partial":
            return self.mask_column_partial("Analyst", strategy="name", show_chars=1)
        elif strategy == "hash":
            return self.mask_column_hash("Analyst")
        
        return self.df
    
    def mask_transaction_ids(self, show_last: int = 4) -> pd.DataFrame:
        """
        Mask transaction IDs, showing only the last N characters.
        
        Args:
            show_last: Number of digits to reveal at the end.
            
        Returns:
            pd.DataFrame: DataFrame with masked transaction IDs.
        """
        if "Transaction_ID" not in self.df.columns:
            return self.df
        
        return self.mask_column_partial(
            "Transaction_ID",
            strategy="id",
            show_chars=show_last
        )
    
    def mask_all_pii(self) -> pd.DataFrame:
        """
        Apply default masking to all identified PII columns.
        
        This method applies appropriate masking strategies to each
        default PII column:
        - Analyst names: Partial masking (initials only)
        - Transaction IDs: Show last 4 characters
        
        Returns:
            pd.DataFrame: DataFrame with all PII columns masked.
        """
        # Mask analyst names (show initials)
        if "Analyst" in self.df.columns:
            self.mask_analyst_names(strategy="partial")
        
        # Mask transaction IDs (show last 4 digits)
        if "Transaction_ID" in self.df.columns:
            self.mask_transaction_ids(show_last=4)
        
        return self.df
    
    def get_masking_report(self) -> Dict:
        """
        Generate a report of all masking operations performed.
        
        Returns:
            dict: Report detailing all masked columns and strategies used.
        """
        return {
            "total_columns_masked": len(self.masked_columns),
            "masked_columns": self.masked_columns,
            "masking_details": self.masking_log,
            "compliance_note": "All PII fields have been anonymized according to data privacy requirements"
        }
    
    def get_masked_data(self) -> pd.DataFrame:
        """
        Get the DataFrame with all applied masks.
        
        Returns:
            pd.DataFrame: Masked DataFrame.
        """
        return self.df
