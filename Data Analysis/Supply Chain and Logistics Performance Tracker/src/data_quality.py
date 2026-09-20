"""
Data Quality and Anonymization Engine for Supply Chain Analytics
Handles validation, cleaning, and privacy compliance for shipment data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import hashlib


class DataQualityValidator:
    """Validates and cleans supply chain shipment data."""
    
    REQUIRED_COLUMNS = [
        'Shipment_ID', 'Date', 'Year', 'Month', 'Quarter', 'Carrier',
        'Origin', 'Destination', 'Cargo_Type', 'Mode', 'Client_Type',
        'Incoterm', 'Warehouse', 'Analyst', 'Planned_Days', 'Actual_Days',
        'Delay_Days', 'On_Time', 'Cargo_Value', 'Freight_Cost', 'SLA_Penalty',
        'Customs_Hold', 'Delivered'
    ]
    
    VALID_CARGO_TYPES = ['Perishables', 'Pharma', 'Automotive', 'Electronics', 
                         'Textiles', 'Machinery', 'Chemicals', 'Consumer Goods']
    VALID_MODES = ['Air Freight', 'Sea Freight', 'Road', 'Rail']
    VALID_INCOTERMS = ['FOB', 'CIF', 'EXW', 'DDP', 'DAP']
    
    def __init__(self, tolerance_days: int = 1):
        self.tolerance_days = tolerance_days
        self.validation_errors: Dict[str, Any] = {}
        
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """Check if all required columns are present."""
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            self.validation_errors['missing_columns'] = missing_cols
            return False, missing_cols
        return True, []
    
    def validate_data_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate business rules and data integrity."""
        errors = []
        
        # Actual_Days cannot be less than Planned_Days minus tolerance
        invalid_days = df[df['Actual_Days'] < df['Planned_Days'] - self.tolerance_days]
        if len(invalid_days) > 0:
            errors.append(f"Found {len(invalid_days)} records with Actual_Days < Planned_Days - tolerance")
            # Fix: set Actual_Days to Planned_Days for these records
            df.loc[invalid_days.index, 'Actual_Days'] = df.loc[invalid_days.index, 'Planned_Days']
        
        # Delay_Days should equal Actual_Days - Planned_Days (when positive)
        expected_delay = (df['Actual_Days'] - df['Planned_Days']).clip(lower=0)
        mismatched_delay = df[df['Delay_Days'] != expected_delay]
        if len(mismatched_delay) > 0:
            errors.append(f"Found {len(mismatched_delay)} records with incorrect Delay_Days calculation")
            df['Delay_Days'] = expected_delay
        
        # On_Time should be 1 when Delay_Days == 0, else 0
        expected_on_time = (df['Delay_Days'] == 0).astype(int)
        mismatched_on_time = df[df['On_Time'] != expected_on_time]
        if len(mismatched_on_time) > 0:
            errors.append(f"Found {len(mismatched_on_time)} records with incorrect On_Time flag")
            df['On_Time'] = expected_on_time
        
        # Cargo_Value must be positive
        invalid_value = df[df['Cargo_Value'] <= 0]
        if len(invalid_value) > 0:
            errors.append(f"Found {len(invalid_value)} records with non-positive Cargo_Value")
        
        # Freight_Cost must be positive
        invalid_cost = df[df['Freight_Cost'] <= 0]
        if len(invalid_cost) > 0:
            errors.append(f"Found {len(invalid_cost)} records with non-positive Freight_Cost")
        
        self.validation_errors['integrity_checks'] = errors
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Handle missing values with appropriate strategies."""
        missing_counts = df.isnull().sum().to_dict()
        missing_counts = {k: v for k, v in missing_counts.items() if v > 0}
        
        # Numeric columns: fill with median
        numeric_cols = ['Planned_Days', 'Actual_Days', 'Delay_Days', 'Cargo_Value', 
                        'Freight_Cost', 'SLA_Penalty']
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Categorical columns: fill with mode
        categorical_cols = ['Carrier', 'Cargo_Type', 'Mode', 'Client_Type', 
                           'Incoterm', 'Warehouse', 'Analyst']
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
        
        # Binary columns: fill with 0
        binary_cols = ['On_Time', 'Customs_Hold', 'Delivered']
        for col in binary_cols:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(0, inplace=True)
        
        return df, missing_counts
    
    def validate_and_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run full validation and cleaning pipeline."""
        report = {
            'original_rows': len(df),
            'validation_passed': True,
            'issues': {}
        }
        
        # Schema validation
        schema_valid, missing_cols = self.validate_schema(df)
        if not schema_valid:
            report['validation_passed'] = False
            report['issues']['schema'] = missing_cols
            raise ValueError(f"Schema validation failed. Missing columns: {missing_cols}")
        
        # Handle missing values
        cleaned_df, missing_info = self.handle_missing_values(df.copy())
        report['issues']['missing_values'] = missing_info
        
        # Data integrity validation
        cleaned_df = self.validate_data_integrity(cleaned_df)
        report['issues']['integrity'] = self.validation_errors.get('integrity_checks', [])
        
        report['cleaned_rows'] = len(cleaned_df)
        report['duplicate_rows_removed'] = report['original_rows'] - report['cleaned_rows']
        
        return cleaned_df, report


class DataAnonymizer:
    """Anonymizes sensitive data fields for privacy compliance."""
    
    def __init__(self, salt: str = "supply_chain_2024"):
        self.salt = salt
        
    def hash_value(self, value: str) -> str:
        """Create a deterministic hash of a value."""
        if pd.isna(value) or value is None:
            return "ANONYMIZED"
        combined = f"{self.salt}_{value}"
        return hashlib.sha256(combined.encode()).hexdigest()[:12]
    
    def anonymize_analyst(self, df: pd.DataFrame, column: str = 'Analyst') -> pd.DataFrame:
        """Anonymize the Analyst column using hashing."""
        df_copy = df.copy()
        if column in df_copy.columns:
            unique_analysts = df_copy[column].unique()
            mapping = {analyst: f"Analyst_{i+1}" for i, analyst in enumerate(sorted(unique_analysts))}
            df_copy[column] = df_copy[column].map(mapping)
        return df_copy
    
    def anonymize_client(self, df: pd.DataFrame, column: str = 'Client_Type') -> pd.DataFrame:
        """Generalize client types for privacy."""
        df_copy = df.copy()
        if column in df_copy.columns:
            # Keep client type categories but remove specific identifiers
            df_copy[column] = df_copy[column].apply(lambda x: x if x in ['Enterprise', 'SMB', 'Government', 'Startup'] else 'Other')
        return df_copy
    
    def create_anonymized_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a fully anonymized version of the dataset."""
        df_anon = df.copy()
        df_anon = self.anonymize_analyst(df_anon)
        df_anon = self.anonymize_client(df_anon)
        
        # Remove or hash any potentially identifying information
        if 'Shipment_ID' in df_anon.columns:
            df_anon['Shipment_ID'] = df_anon['Shipment_ID'].apply(self.hash_value)
        
        return df_anon


def load_and_validate_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load data from CSV and run validation pipeline."""
    df = pd.read_csv(file_path)
    validator = DataQualityValidator()
    cleaned_df, report = validator.validate_and_clean(df)
    return cleaned_df, report
