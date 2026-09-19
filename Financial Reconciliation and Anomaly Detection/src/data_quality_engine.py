"""
Data Quality Engine Module
==========================

Provides comprehensive data validation, schema checking, missing value handling,
and data quality scoring for financial transaction data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class DataQualityEngine:
    """
    Comprehensive data quality validation engine for financial transactions.
    
    This class implements:
    - Schema validation (column names, types)
    - Missing value detection and handling
    - Duplicate identification
    - Format consistency checks
    - Data quality scoring
    
    Attributes:
        df (pd.DataFrame): Input DataFrame to validate.
        expected_schema (dict): Expected column names and types.
        validation_results (dict): Storage for validation results.
    """
    
    # Define expected schema for financial transaction data
    EXPECTED_SCHEMA = {
        "Transaction_ID": "string",
        "Date": "datetime64[ns]",
        "Year": "Int64",
        "Month": "string",
        "Quarter": "string",
        "Department": "string",
        "Category": "string",
        "Transaction_Type": "string",
        "Source_System": "string",
        "Region": "string",
        "Currency": "string",
        "Analyst": "string",
        "Expected_Amount": "float64",
        "Actual_Amount": "float64",
        "Discrepancy": "float64",
        "Discrepancy_Pct": "float64",
        "Status": "string"
    }
    
    # Columns that must not have missing values
    CRITICAL_COLUMNS = [
        "Transaction_ID", "Date", "Department", 
        "Expected_Amount", "Actual_Amount", "Status"
    ]
    
    # Valid status values
    VALID_STATUSES = ["Reconciled", "Under Review", "Flagged", "Pending"]
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Data Quality Engine.
        
        Args:
            df: DataFrame containing transaction data to validate.
        """
        self.df = df.copy()
        self.validation_results: Dict = {}
        self.cleaned_df: pd.DataFrame = df.copy()
    
    def validate_schema(self) -> Dict:
        """
        Validate that the DataFrame has the expected schema.
        
        Returns:
            dict: Schema validation results including missing columns,
                  extra columns, and type mismatches.
        """
        actual_columns = set(self.df.columns)
        expected_columns = set(self.EXPECTED_SCHEMA.keys())
        
        missing_columns = list(expected_columns - actual_columns)
        extra_columns = list(actual_columns - expected_columns)
        
        # Check data types where possible
        type_mismatches = []
        for col in self.df.columns:
            if col in self.EXPECTED_SCHEMA:
                actual_dtype = str(self.df[col].dtype)
                expected_dtype = self.EXPECTED_SCHEMA[col]
                # Flexible type checking
                if not self._check_type_compatible(actual_dtype, expected_dtype):
                    type_mismatches.append({
                        "column": col,
                        "expected": expected_dtype,
                        "actual": actual_dtype
                    })
        
        result = {
            "schema_valid": len(missing_columns) == 0,
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
            "type_mismatches": type_mismatches,
            "timestamp": datetime.now().isoformat()
        }
        
        self.validation_results["schema"] = result
        return result
    
    def _check_type_compatible(self, actual: str, expected: str) -> bool:
        """
        Check if actual dtype is compatible with expected dtype.
        
        Args:
            actual: Actual dtype string.
            expected: Expected dtype string.
            
        Returns:
            bool: True if compatible, False otherwise.
        """
        # Handle nullable integer types
        if expected == "Int64" and actual in ["Int64", "int64", "float64"]:
            return True
        # Handle float types
        if expected == "float64" and actual in ["float64", "Int64", "int64"]:
            return True
        # Handle datetime
        if expected == "datetime64[ns]" and "datetime" in actual:
            return True
        # Handle string types
        if expected == "string" and actual in ["string", "object"]:
            return True
        
        return actual == expected
    
    def check_missing_values(self) -> Dict:
        """
        Identify and report missing values in the dataset.
        
        Returns:
            dict: Missing value statistics per column and overall.
        """
        missing_counts = self.df.isnull().sum()
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).round(2)
        
        # Identify critical columns with missing values
        critical_missing = {}
        for col in self.CRITICAL_COLUMNS:
            if col in self.df.columns:
                count = missing_counts.get(col, 0)
                if count > 0:
                    critical_missing[col] = {
                        "count": int(count),
                        "percentage": float(missing_pct.get(col, 0))
                    }
        
        result = {
            "total_missing": int(missing_counts.sum()),
            "missing_by_column": missing_counts.to_dict(),
            "missing_percentage_by_column": missing_pct.to_dict(),
            "critical_columns_with_missing": critical_missing,
            "has_critical_missing": len(critical_missing) > 0,
            "timestamp": datetime.now().isoformat()
        }
        
        self.validation_results["missing_values"] = result
        return result
    
    def identify_duplicates(self) -> Dict:
        """
        Identify duplicate records in the dataset.
        
        Checks for:
        - Exact row duplicates
        - Duplicate Transaction_IDs
        
        Returns:
            dict: Duplicate detection results.
        """
        # Check for exact duplicates
        exact_duplicates = self.df.duplicated().sum()
        
        # Check for duplicate Transaction_IDs
        if "Transaction_ID" in self.df.columns:
            duplicate_txn_ids = self.df[self.df.duplicated(subset=["Transaction_ID"], keep=False)]
            duplicate_txn_count = len(duplicate_txn_ids)
            unique_txn_ids = self.df["Transaction_ID"].duplicated().sum()
        else:
            duplicate_txn_count = 0
            unique_txn_ids = 0
        
        result = {
            "duplicate_count": int(exact_duplicates),
            "duplicate_transaction_ids": int(unique_txn_ids),
            "duplicate_transaction_id_records": duplicate_txn_count,
            "has_duplicates": exact_duplicates > 0 or unique_txn_ids > 0,
            "timestamp": datetime.now().isoformat()
        }
        
        self.validation_results["duplicates"] = result
        return result
    
    def check_format_consistency(self) -> Dict:
        """
        Check for format inconsistencies in key fields.
        
        Validates:
        - Transaction_ID format (should be TXN-XXXXXX)
        - Date format and range
        - Numeric field validity
        - Status value validity
        
        Returns:
            dict: Format consistency check results.
        """
        issues = []
        
        # Check Transaction_ID format
        if "Transaction_ID" in self.df.columns:
            invalid_txn_format = self.df[
                ~self.df["Transaction_ID"].str.match(r"TXN-\d+", na=False)
            ]
            if len(invalid_txn_format) > 0:
                issues.append({
                    "field": "Transaction_ID",
                    "issue": "Invalid format",
                    "count": len(invalid_txn_format)
                })
        
        # Check Date range (should be within reasonable bounds)
        if "Date" in self.df.columns:
            min_date = self.df["Date"].min()
            max_date = self.df["Date"].max()
            # Check for dates before 2000 or in the future
            invalid_dates = self.df[
                (self.df["Date"] < pd.Timestamp("2000-01-01")) |
                (self.df["Date"] > pd.Timestamp.now())
            ]
            if len(invalid_dates) > 0:
                issues.append({
                    "field": "Date",
                    "issue": "Date out of valid range",
                    "count": len(invalid_dates),
                    "min_found": str(min_date),
                    "max_found": str(max_date)
                })
        
        # Check Status values
        if "Status" in self.df.columns:
            invalid_statuses = self.df[~self.df["Status"].isin(self.VALID_STATUSES)]
            if len(invalid_statuses) > 0:
                unique_invalid = invalid_statuses["Status"].unique().tolist()
                issues.append({
                    "field": "Status",
                    "issue": "Invalid status values",
                    "count": len(invalid_statuses),
                    "unique_values": unique_invalid
                })
        
        # Check numeric fields for negative values where inappropriate
        numeric_fields = ["Expected_Amount", "Actual_Amount"]
        for field in numeric_fields:
            if field in self.df.columns:
                negative_count = (self.df[field] < 0).sum()
                if negative_count > 0:
                    issues.append({
                        "field": field,
                        "issue": "Negative values found",
                        "count": int(negative_count)
                    })
        
        result = {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "total_issues": len(issues),
            "timestamp": datetime.now().isoformat()
        }
        
        self.validation_results["format_consistency"] = result
        return result
    
    def handle_missing_values(self, strategy: str = "drop_critical") -> pd.DataFrame:
        """
        Handle missing values based on specified strategy.
        
        Args:
            strategy: One of:
                - "drop_critical": Drop rows with missing critical columns
                - "drop_all": Drop any row with missing values
                - "fill_numeric_mean": Fill numeric with mean, drop critical
                - "none": Do not handle missing values
                
        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        df = self.cleaned_df.copy()
        
        if strategy == "drop_critical":
            # Drop rows where critical columns are missing
            df = df.dropna(subset=self.CRITICAL_COLUMNS)
        elif strategy == "drop_all":
            df = df.dropna()
        elif strategy == "fill_numeric_mean":
            # Fill numeric columns with mean
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
            # Drop rows with missing critical non-numeric columns
            df = df.dropna(subset=self.CRITICAL_COLUMNS)
        
        self.cleaned_df = df
        return df
    
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate records from the DataFrame.
        
        Args:
            subset: Columns to consider for duplicate detection.
                   If None, uses all columns.
                   
        Returns:
            pd.DataFrame: DataFrame with duplicates removed.
        """
        if subset is None:
            subset = ["Transaction_ID"] if "Transaction_ID" in self.cleaned_df.columns else None
        
        self.cleaned_df = self.cleaned_df.drop_duplicates(subset=subset, keep="first")
        return self.cleaned_df
    
    def calculate_quality_score(self) -> float:
        """
        Calculate an overall data quality score (0 to 1).
        
        The score is based on:
        - Schema validity (25%)
        - Missing value ratio (25%)
        - Duplicate ratio (25%)
        - Format consistency (25%)
        
        Returns:
            float: Quality score between 0 and 1.
        """
        # Ensure all validations have been run
        if "schema" not in self.validation_results:
            self.validate_schema()
        if "missing_values" not in self.validation_results:
            self.check_missing_values()
        if "duplicates" not in self.validation_results:
            self.identify_duplicates()
        if "format_consistency" not in self.validation_results:
            self.check_format_consistency()
        
        # Schema score (0 or 1)
        schema_score = 1.0 if self.validation_results["schema"]["schema_valid"] else 0.5
        
        # Missing value score (based on percentage of non-missing)
        total_cells = len(self.df) * len(self.df.columns)
        missing_cells = self.validation_results["missing_values"]["total_missing"]
        missing_score = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 1.0
        
        # Duplicate score
        dup_count = self.validation_results["duplicates"]["duplicate_count"]
        dup_score = 1.0 - (dup_count / len(self.df)) if len(self.df) > 0 else 1.0
        
        # Format consistency score
        format_issues = self.validation_results["format_consistency"]["total_issues"]
        format_score = max(0, 1.0 - (format_issues * 0.1))
        
        # Weighted average
        quality_score = (
            0.25 * schema_score +
            0.25 * missing_score +
            0.25 * dup_score +
            0.25 * format_score
        )
        
        return quality_score
    
    def run_full_validation(self) -> Dict:
        """
        Run all validation checks and compile a comprehensive report.
        
        Returns:
            dict: Complete validation report with all checks and quality score.
        """
        self.validate_schema()
        self.check_missing_values()
        self.identify_duplicates()
        self.check_format_consistency()
        
        quality_score = self.calculate_quality_score()
        
        report = {
            "schema_valid": self.validation_results["schema"]["schema_valid"],
            "total_missing": self.validation_results["missing_values"]["total_missing"],
            "duplicate_count": self.validation_results["duplicates"]["duplicate_count"],
            "format_issues_count": self.validation_results["format_consistency"]["total_issues"],
            "quality_score": quality_score,
            "detailed_results": self.validation_results,
            "record_count": len(self.df),
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Get the cleaned DataFrame after applying quality fixes.
        
        Returns:
            pd.DataFrame: Cleaned transaction data.
        """
        # Apply default cleaning: drop critical missing and duplicates
        self.handle_missing_values(strategy="drop_critical")
        self.remove_duplicates()
        return self.cleaned_df
