"""
Data Validation Module

Provides data quality validation including missing value checks,
schema validation, and data type enforcement.
"""

import pandas as pd
from typing import Dict, List, Any


class DataValidator:
    """
    Validates HR data for quality issues including missing values,
    incorrect data types, and schema compliance.
    """
    
    REQUIRED_COLUMNS = [
        'Age', 'Attrition', 'BusinessTravel', 'Department',
        'EmployeeNumber', 'Gender', 'JobRole', 'MonthlyIncome',
        'OverTime', 'TotalWorkingYears', 'YearsAtCompany'
    ]
    
    CATEGORICAL_COLUMNS = [
        'Attrition', 'BusinessTravel', 'Department', 'EducationField',
        'Gender', 'JobRole', 'MaritalStatus', 'OverTime', 'Over18'
    ]
    
    NUMERICAL_COLUMNS = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education',
        'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
        'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
        'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
        'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ]
    
    VALID_ATTRITION_VALUES = ['Yes', 'No']
    VALID_OVERTIME_VALUES = ['Yes', 'No']
    
    def __init__(self):
        """Initialize the DataValidator."""
        self.validation_report: Dict[str, Any] = {}
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive validation on the DataFrame.
        
        Args:
            df: The DataFrame to validate.
            
        Returns:
            dict: Validation report with status and details.
        """
        self.validation_report = {
            'status': 'passed',
            'missing_columns': [],
            'missing_values': {},
            'invalid_values': {},
            'duplicate_rows': 0,
            'issues': []
        }
        
        # Check required columns
        self._check_required_columns(df)
        
        # Check missing values
        self._check_missing_values(df)
        
        # Check for duplicates
        self._check_duplicates(df)
        
        # Validate categorical values
        self._validate_categorical_values(df)
        
        # Validate numerical ranges
        self._validate_numerical_ranges(df)
        
        # Update overall status
        if (self.validation_report['missing_columns'] or 
            self.validation_report['invalid_values'] or
            self.validation_report['issues']):
            self.validation_report['status'] = 'warning'
        
        return self.validation_report
    
    def _check_required_columns(self, df: pd.DataFrame) -> None:
        """Check if all required columns are present."""
        actual_columns = set(df.columns)
        missing = [col for col in self.REQUIRED_COLUMNS if col not in actual_columns]
        self.validation_report['missing_columns'] = missing
        
        if missing:
            self.validation_report['issues'].append(
                f"Missing required columns: {missing}"
            )
    
    def _check_missing_values(self, df: pd.DataFrame) -> None:
        """Identify columns with missing values."""
        missing_counts = df.isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0]
        
        self.validation_report['missing_values'] = columns_with_missing.to_dict()
        
        if columns_with_missing.any():
            for col, count in columns_with_missing.items():
                self.validation_report['issues'].append(
                    f"Column '{col}' has {count} missing values ({count/len(df)*100:.1f}%)"
                )
    
    def _check_duplicates(self, df: pd.DataFrame) -> None:
        """Check for duplicate rows based on EmployeeNumber."""
        if 'EmployeeNumber' in df.columns:
            duplicates = df.duplicated(subset=['EmployeeNumber'], keep=False).sum()
            self.validation_report['duplicate_rows'] = duplicates
            
            if duplicates > 0:
                self.validation_report['issues'].append(
                    f"Found {duplicates} duplicate rows based on EmployeeNumber"
                )
        else:
            # Check full row duplicates
            duplicates = df.duplicated().sum()
            self.validation_report['duplicate_rows'] = duplicates
    
    def _validate_categorical_values(self, df: pd.DataFrame) -> None:
        """Validate that categorical columns have expected values."""
        # Check Attrition column
        if 'Attrition' in df.columns:
            invalid_attrition = df[~df['Attrition'].isin(self.VALID_ATTRITION_VALUES)]
            if len(invalid_attrition) > 0:
                self.validation_report['invalid_values']['Attrition'] = list(
                    df['Attrition'].unique()
                )
                self.validation_report['issues'].append(
                    f"Invalid values in Attrition column: {df['Attrition'].unique()}"
                )
        
        # Check OverTime column
        if 'OverTime' in df.columns:
            invalid_overtime = df[~df['OverTime'].isin(self.VALID_OVERTIME_VALUES)]
            if len(invalid_overtime) > 0:
                self.validation_report['invalid_values']['OverTime'] = list(
                    df['OverTime'].unique()
                )
    
    def _validate_numerical_ranges(self, df: pd.DataFrame) -> None:
        """Validate numerical columns are within reasonable ranges."""
        # Age should be positive and reasonable (18-100)
        if 'Age' in df.columns:
            invalid_age = df[(df['Age'] < 18) | (df['Age'] > 100)]
            if len(invalid_age) > 0:
                self.validation_report['issues'].append(
                    f"Found {len(invalid_age)} records with invalid Age values"
                )
        
        # MonthlyIncome should be positive
        if 'MonthlyIncome' in df.columns:
            invalid_income = df[df['MonthlyIncome'] <= 0]
            if len(invalid_income) > 0:
                self.validation_report['issues'].append(
                    f"Found {len(invalid_income)} records with non-positive MonthlyIncome"
                )
        
        # YearsAtCompany should be non-negative
        if 'YearsAtCompany' in df.columns:
            invalid_tenure = df[df['YearsAtCompany'] < 0]
            if len(invalid_tenure) > 0:
                self.validation_report['issues'].append(
                    f"Found {len(invalid_tenure)} records with negative YearsAtCompany"
                )
    
    def handle_missing_values(self, df: pd.DataFrame, 
                              strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: The DataFrame with missing values.
            strategy: Strategy for handling missing values ('median', 'mean', 'mode', 'drop').
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        df_clean = df.copy()
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        elif strategy == 'median':
            for col in df_clean.select_dtypes(include=['float64', 'int64']).columns:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            for col in df_clean.select_dtypes(include=['object', 'category']).columns:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        elif strategy == 'mean':
            for col in df_clean.select_dtypes(include=['float64', 'int64']).columns:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            for col in df_clean.select_dtypes(include=['object', 'category']).columns:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        elif strategy == 'mode':
            for col in df_clean.columns:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        return df_clean
