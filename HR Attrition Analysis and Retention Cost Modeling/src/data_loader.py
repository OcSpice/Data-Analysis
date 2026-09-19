"""
Data Loading Module

Handles loading of HR attrition data from CSV files with proper error handling.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class DataLoader:
    """Responsible for loading and initial inspection of HR datasets."""
    
    EXPECTED_COLUMNS = [
        'Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
        'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
        'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
        'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
        'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
        'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ]
    
    def __init__(self, file_path: str):
        """
        Initialize the DataLoader.
        
        Args:
            file_path: Path to the CSV file containing HR data.
        """
        self.file_path = Path(file_path)
        self.data: Optional[pd.DataFrame] = None
    
    def load(self) -> pd.DataFrame:
        """
        Load the CSV file into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Loaded dataset.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is empty or has no columns.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        self.data = pd.read_csv(self.file_path)
        
        if self.data.empty:
            raise ValueError("Loaded data is empty")
        
        if len(self.data.columns) == 0:
            raise ValueError("Loaded data has no columns")
        
        return self.data
    
    def get_summary(self) -> dict:
        """
        Get a summary of the loaded data.
        
        Returns:
            dict: Summary statistics including row count, column count, and memory usage.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        return {
            'row_count': len(self.data),
            'column_count': len(self.data.columns),
            'memory_usage_bytes': self.data.memory_usage(deep=True).sum(),
            'columns': list(self.data.columns)
        }
    
    def validate_schema(self) -> tuple[bool, list]:
        """
        Check if all expected columns are present.
        
        Returns:
            tuple: (is_valid, missing_columns)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        actual_columns = set(self.data.columns)
        expected_set = set(self.EXPECTED_COLUMNS)
        
        missing = list(expected_set - actual_columns)
        is_valid = len(missing) == 0
        
        return is_valid, missing
