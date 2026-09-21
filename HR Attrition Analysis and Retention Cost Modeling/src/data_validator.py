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
        """Perform comprehensive validation on the DataFrame."""
        self.validation_report = {
            'status': 'passed',
            'missing_columns': [],
            'missing_values': {},
            'invalid_values': {},
            'duplicate_rows': 0,
            'issues': []
        }
        
        self._check_required_columns(df)
        self._check_missing_values(df)
        self._check_duplicates(df)
        self._validate_categorical_values(df)
        self._validate_numerical_ranges(df)
        
        if (
            self.validation_report['missing_columns']
            or self.validation_report['invalid_values']
            or self.validation_report['issues']
        ):
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
            duplicates = df.duplicated().sum()
            self.validation_report['duplicate_rows'] = duplicates
    
    def _validate_categorical_values(self, df: pd.DataFrame) -> None:
        """Validate that categorical columns have expected values."""
        if 'Attrition' in df.columns:
            invalid_attrition = df[~df['Attrition'].isin(self.VALID_ATTRITION_VALUES)]
            if len(invalid_attrition) > 0:
                self.validation_report['invalid_values']['Attrition'] = list(
                    df['Attrition'].unique()
                )
                self.validation_report['issues'].append(
                    f"Invalid values in Attrition column: {df['Attrition'].unique()}"
                )
        
        if 'OverTime' in df.columns:
            invalid_overtime = df[~df['OverTime'].isin(self.VALID_OVERTIME_VALUES)]
            if len(invalid_overtime) > 0:
                self.validation_report['invalid_values']['OverTime'] = list(
                    df['OverTime'].unique()
                )
    
    def _validate_numerical_ranges(self, df: pd.DataFrame) -> None:
        """Validate numerical columns are within reasonable ranges."""
        if 'Age' in df.columns:
            invalid_age = df[(df['Age'] < 18) | (df['Age'] > 100)]
            if len(invalid_age) > 0:
                self.validation_report['issues'].append(
                    f"Found {len(invalid_age)} records with invalid Age values"
                )
        
        if 'MonthlyIncome' in df.columns:
            invalid_income = df[df['MonthlyIncome'] <= 0]
            if len(invalid_income) > 0:
                self.validation_report['issues'].append(
                    f"Found {len(invalid_income)} records with non-positive MonthlyIncome"
                )
        
        if 'YearsAtCompany' in df.columns:
            invalid_tenure = df[df['YearsAtCompany'] < 0]
            if len(invalid_tenure) > 0:
                self.validation_report['issues'].append(
                    f"Found {len(invalid_tenure)} records with negative YearsAtCompany"
                )
    
    def handle_missing_values(
        self, df: pd.DataFrame, strategy: str = 'median'
    ) -> pd.DataFrame:
        """
        Handle missing values without chained inplace assignment.

        The predictive modeling pipeline performs its own train-only
        imputation. This helper remains available for standalone data-quality
        workflows and tests.
        """
        if strategy not in {'median', 'mean', 'mode', 'drop'}:
            raise ValueError(
                "strategy must be one of: 'median', 'mean', 'mode', 'drop'"
            )

        df_clean = df.copy()
        
        if strategy == 'drop':
            return df_clean.dropna()

        if strategy in {'median', 'mean'}:
            numeric_columns = df_clean.select_dtypes(
                include=['number']
            ).columns
            categorical_columns = df_clean.select_dtypes(
                include=['object', 'category', 'string']
            ).columns

            for col in numeric_columns:
                fill_value = (
                    df_clean[col].median()
                    if strategy == 'median'
                    else df_clean[col].mean()
                )
                if pd.notna(fill_value):
                    df_clean[col] = df_clean[col].fillna(fill_value)

            for col in categorical_columns:
                mode = df_clean[col].mode(dropna=True)
                if not mode.empty:
                    df_clean[col] = df_clean[col].fillna(mode.iloc[0])

            return df_clean

        for col in df_clean.columns:
            mode = df_clean[col].mode(dropna=True)
            if not mode.empty:
                df_clean[col] = df_clean[col].fillna(mode.iloc[0])

        return df_clean
