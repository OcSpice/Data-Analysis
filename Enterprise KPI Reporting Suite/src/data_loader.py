"""
Data Loading Module for Enterprise KPI Reporting Suite
Handles CSV loading, schema validation, and data quality checks.

Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and initial validation of enterprise KPI data.
    
    Attributes:
        data_path (Path): Path to the CSV data file
        df (pd.DataFrame): Loaded dataframe
    """
    
    EXPECTED_COLUMNS = [
        'Record_ID', 'Date', 'Year', 'Month', 'Quarter', 'Department', 
        'Region', 'Product', 'Channel', 'Employee_ID', 'Cost_Center',
        'Revenue', 'Cost', 'Gross_Margin', 'Margin_Pct', 'Deals_Closed',
        'Leads_Generated', 'Conv_Rate_Pct', 'Customer_Sat', 'NPS',
        'Headcount', 'Attrition_Pct', 'Training_Hrs', 'Productivity_Pct',
        'CAC', 'LTV', 'LTV_CAC_Ratio', 'MQL_Count', 'SQL_Count', 'CTR_Pct',
        'SLA_Met', 'Ticket_Volume', 'Resolution_Hrs', 'Uptime_Pct', 'Status'
    ]
    
    def __init__(self, data_path: str):
        """
        Initialize the DataLoader with a path to the data file.
        
        Args:
            data_path: Path to the CSV file containing KPI data
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.validation_errors: List[str] = []
    
    def load(self) -> pd.DataFrame:
        """
        Load the CSV file into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: The loaded dataframe
            
        Raises:
            FileNotFoundError: If the data file does not exist
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} records")
        
        return self.df
    
    def validate_schema(self) -> Tuple[bool, List[str]]:
        """
        Validate that all expected columns are present in the dataframe.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of missing columns)
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")
        
        missing_cols = set(self.EXPECTED_COLUMNS) - set(self.df.columns)
        is_valid = len(missing_cols) == 0
        
        if not is_valid:
            self.validation_errors.append(f"Missing columns: {missing_cols}")
            logger.error(f"Schema validation failed. Missing: {missing_cols}")
        else:
            logger.info("Schema validation passed")
        
        return is_valid, list(missing_cols)
    
    def check_data_quality(self) -> dict:
        """
        Perform comprehensive data quality checks.
        
        Returns:
            dict: Quality report with metrics and issues
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")
        
        quality_report = {
            'total_records': len(self.df),
            'missing_values': {},
            'duplicate_records': 0,
            'invalid_revenue_cost': 0,
            'negative_values': {},
            'quality_score': 100.0
        }
        
        # Check missing values
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                quality_report['missing_values'][col] = int(missing_count)
        
        # Check for duplicates
        quality_report['duplicate_records'] = int(self.df.duplicated().sum())
        
        # Check Revenue >= Cost constraint
        if 'Revenue' in self.df.columns and 'Cost' in self.df.columns:
            invalid_rc = (self.df['Revenue'] < self.df['Cost']).sum()
            quality_report['invalid_revenue_cost'] = int(invalid_rc)
            if invalid_rc > 0:
                self.validation_errors.append(f"{invalid_rc} records have Revenue < Cost")
        
        # Check for negative values in key numeric columns
        numeric_cols = ['Revenue', 'Cost', 'Gross_Margin', 'Headcount', 'Deals_Closed']
        for col in numeric_cols:
            if col in self.df.columns:
                neg_count = (self.df[col] < 0).sum()
                if neg_count > 0:
                    quality_report['negative_values'][col] = int(neg_count)
        
        # Calculate quality score
        total_checks = len(self.df) * len(self.df.columns)
        total_issues = sum(quality_report['missing_values'].values())
        total_issues += quality_report['duplicate_records']
        total_issues += quality_report['invalid_revenue_cost']
        
        quality_report['quality_score'] = round(
            max(0, 100 - (total_issues / total_checks * 100)), 2
        )
        
        logger.info(f"Data quality score: {quality_report['quality_score']}%")
        
        return quality_report
    
    def handle_missing_values(self, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the dataframe.
        
        Args:
            strategy: One of 'drop', 'fill_mean', 'fill_median', 'fill_zero'
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")
        
        original_count = len(self.df)
        
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill_mean':
            numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'fill_median':
            numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'fill_zero':
            self.df = self.df.fillna(0)
        
        dropped_count = original_count - len(self.df)
        logger.info(f"Handled missing values using '{strategy}' strategy. Dropped {dropped_count} records.")
        
        return self.df
