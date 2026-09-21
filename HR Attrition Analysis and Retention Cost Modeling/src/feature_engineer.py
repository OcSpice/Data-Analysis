"""
Feature Engineering Module

Creates derived features for attrition analysis including
replacement cost calculations and tenure buckets.
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineer:
    """
    Engineers features for HR attrition analysis including
    financial metrics, tenure categorizations, and risk indicators.
    """
    
    # Replacement-cost assumption configured by the analysis
    DEFAULT_REPLACEMENT_COST_MULTIPLIER = 1.5
    
    # Tenure bucket boundaries in years
    TENURE_BUCKETS = [0, 1, 3, 5, 10, 20, 50]
    TENURE_LABELS = ['0-1yr', '1-3yr', '3-5yr', '5-10yr', '10-20yr', '20+yr']
    
    def __init__(self, replacement_cost_multiplier: float = 1.5):
        """
        Initialize the FeatureEngineer.
        
        Args:
            replacement_cost_multiplier: Multiplier for calculating replacement cost
                                         based on annual salary.
        """
        self.replacement_cost_multiplier = replacement_cost_multiplier
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features for the dataset.
        
        Args:
            df: Raw DataFrame with HR data.
            
        Returns:
            pd.DataFrame: DataFrame with additional engineered features.
        """
        df_engineered = df.copy()
        
        # Financial features
        df_engineered = self._create_financial_features(df_engineered)
        
        # Tenure features
        df_engineered = self._create_tenure_features(df_engineered)
        
        # Risk indicator features
        df_engineered = self._create_risk_features(df_engineered)
        
        return df_engineered
    
    def _create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create financial-related features.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with financial features added.
        """
        # Annual Income (MonthlyIncome * 12)
        if 'MonthlyIncome' in df.columns:
            df['AnnualIncome'] = df['MonthlyIncome'] * 12
        
        # Replacement Cost (AnnualIncome * multiplier)
        # This is an analytical estimate based on the configured assumption.
        if 'AnnualIncome' in df.columns:
            df['ReplacementCost'] = df['AnnualIncome'] * self.replacement_cost_multiplier
        
        # Income relative to job level (normalized)
        if 'MonthlyIncome' in df.columns and 'JobLevel' in df.columns:
            df['IncomePerJobLevel'] = df['MonthlyIncome'] / df['JobLevel'].replace(0, 1)
        
        return df
    
    def _create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure-related categorical features.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with tenure features added.
        """
        # Tenure Bucket based on YearsAtCompany
        if 'YearsAtCompany' in df.columns:
            df['TenureBucket'] = pd.cut(
                df['YearsAtCompany'],
                bins=self.TENURE_BUCKETS,
                labels=self.TENURE_LABELS,
                include_lowest=True
            )
        
        # Total Experience Bucket based on TotalWorkingYears
        if 'TotalWorkingYears' in df.columns:
            df['TotalExperienceBucket'] = pd.cut(
                df['TotalWorkingYears'],
                bins=[0, 2, 5, 10, 20, 50],
                labels=['Entry (0-2)', 'Junior (2-5)', 'Mid (5-10)', 
                       'Senior (10-20)', 'Executive (20+)'],
                include_lowest=True
            )
        
        # Promotion Stagnation flag (no promotion in last 3 years)
        if 'YearsSinceLastPromotion' in df.columns:
            df['PromotionStagnation'] = (df['YearsSinceLastPromotion'] >= 3).astype(int)
        
        return df
    
    def _create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk indicator features that may correlate with attrition.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with risk features added.
        """
        # Overtime Risk Flag
        if 'OverTime' in df.columns:
            df['IsOverTime'] = (df['OverTime'] == 'Yes').astype(int)
        
        # Recent Hire flag (less than 1 year at company)
        if 'YearsAtCompany' in df.columns:
            df['IsRecentHire'] = (df['YearsAtCompany'] < 1).astype(int)
        
        # Job Hopper flag (worked at many companies relative to experience)
        if 'NumCompaniesWorked' in df.columns and 'TotalWorkingYears' in df.columns:
            df['JobHopperScore'] = df['NumCompaniesWorked'] / df['TotalWorkingYears'].replace(0, 1)
            df['IsJobHopper'] = (df['JobHopperScore'] > 0.5).astype(int)
        
        # Low Satisfaction Composite (multiple low satisfaction scores)
        satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 
                            'RelationshipSatisfaction']
        available_satisfaction = [col for col in satisfaction_cols if col in df.columns]
        if len(available_satisfaction) > 0:
            df['LowSatisfactionCount'] = sum(
                df[col] <= 2 for col in available_satisfaction
            )
            df['HasLowSatisfaction'] = (df['LowSatisfactionCount'] >= 2).astype(int)
        
        # Work-Life Balance Risk
        if 'WorkLifeBalance' in df.columns:
            df['PoorWorkLifeBalance'] = (df['WorkLifeBalance'] <= 2).astype(int)
        
        return df
    
    def calculate_total_replacement_cost(self, df: pd.DataFrame,
                                          attrition_column: str = 'Attrition') -> float:
        """
        Calculate total replacement cost for employees who left.
        
        Args:
            df: DataFrame with ReplacementCost feature.
            attrition_column: Name of the attrition indicator column.
            
        Returns:
            float: Total replacement cost in dollars.
        """
        if 'ReplacementCost' not in df.columns:
            raise ValueError("DataFrame must have 'ReplacementCost' feature")
        
        if attrition_column not in df.columns:
            raise ValueError(f"Column '{attrition_column}' not found")
        
        departed = df[df[attrition_column] == 'Yes']
        total_cost = departed['ReplacementCost'].sum()
        
        return total_cost
    
    def get_feature_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate a summary of engineered features.
        
        Args:
            df: DataFrame with engineered features.
            
        Returns:
            dict: Summary statistics for engineered features.
        """
        engineered_features = [
            'AnnualIncome', 'ReplacementCost', 'TenureBucket',
            'IsOverTime', 'IsRecentHire', 'PromotionStagnation'
        ]
        
        summary = {
            'features_created': [],
            'statistics': {}
        }
        
        for feat in engineered_features:
            if feat in df.columns:
                summary['features_created'].append(feat)
                if df[feat].dtype in ['int64', 'float64']:
                    summary['statistics'][feat] = {
                        'mean': float(df[feat].mean()),
                        'std': float(df[feat].std()),
                        'min': float(df[feat].min()),
                        'max': float(df[feat].max())
                    }
                else:
                    summary['statistics'][feat] = df[feat].value_counts().to_dict()
        
        return summary
