"""
Exploratory Data Analysis Module

Performs comprehensive EDA on HR attrition data including
segmentation analysis and key metric calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class ExploratoryAnalysis:
    """
    Conducts exploratory data analysis on HR attrition datasets,
    focusing on segmentation by key business dimensions.
    """
    
    def __init__(self):
        """Initialize the ExploratoryAnalysis engine."""
        self.analysis_results: Dict[str, Any] = {}
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive exploratory analysis.
        
        Args:
            df: DataFrame with HR data (should include engineered features).
            
        Returns:
            dict: Dictionary containing all analysis results.
        """
        self.analysis_results = {
            'overall_attrition_rate': self._calculate_overall_attrition(df),
            'attrition_by_overtime': self._analyze_by_overtime(df),
            'attrition_by_department': self._analyze_by_department(df),
            'attrition_by_job_role': self._analyze_by_job_role(df),
            'attrition_by_tenure': self._analyze_by_tenure(df),
            'overtime_risk_ratio': self._calculate_overtime_risk_ratio(df),
            'demographic_summary': self._get_demographic_summary(df),
            'financial_exposure': self._calculate_financial_exposure(df)
        }
        
        return self.analysis_results
    
    def _calculate_overall_attrition(self, df: pd.DataFrame) -> float:
        """
        Calculate overall attrition rate.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            float: Overall attrition rate as a decimal.
        """
        if 'Attrition' not in df.columns:
            return 0.0
        
        total_employees = len(df)
        departed = len(df[df['Attrition'] == 'Yes'])
        
        return departed / total_employees if total_employees > 0 else 0.0
    
    def _analyze_by_overtime(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate attrition rates segmented by overtime status.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            dict: Attrition rates for 'Yes' and 'No' overtime categories.
        """
        if 'OverTime' not in df.columns or 'Attrition' not in df.columns:
            return {'Yes': 0.0, 'No': 0.0}
        
        result = {}
        for overtime_status in ['Yes', 'No']:
            subset = df[df['OverTime'] == overtime_status]
            if len(subset) > 0:
                attrition_rate = (subset['Attrition'] == 'Yes').sum() / len(subset)
                result[overtime_status] = attrition_rate
            else:
                result[overtime_status] = 0.0
        
        return result
    
    def _analyze_by_department(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Calculate attrition metrics by department.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            dict: Department-level attrition statistics.
        """
        if 'Department' not in df.columns or 'Attrition' not in df.columns:
            return {}
        
        result = {}
        departments = df['Department'].unique()
        
        for dept in departments:
            subset = df[df['Department'] == dept]
            total = len(subset)
            departed = (subset['Attrition'] == 'Yes').sum()
            avg_income = subset['MonthlyIncome'].mean() if 'MonthlyIncome' in subset.columns else 0
            avg_replacement_cost = subset['ReplacementCost'].mean() if 'ReplacementCost' in subset.columns else 0
            
            result[dept] = {
                'total_employees': int(total),
                'departed_count': int(departed),
                'attrition_rate': float(departed / total) if total > 0 else 0.0,
                'avg_monthly_income': float(avg_income),
                'avg_replacement_cost': float(avg_replacement_cost)
            }
        
        return result
    
    def _analyze_by_job_role(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Calculate attrition metrics by job role.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            dict: Job role-level attrition statistics.
        """
        if 'JobRole' not in df.columns or 'Attrition' not in df.columns:
            return {}
        
        result = {}
        job_roles = df['JobRole'].unique()
        
        for role in job_roles:
            subset = df[df['JobRole'] == role]
            total = len(subset)
            departed = (subset['Attrition'] == 'Yes').sum()
            avg_overtime_rate = subset['IsOverTime'].mean() if 'IsOverTime' in subset.columns else 0
            
            result[role] = {
                'total_employees': int(total),
                'departed_count': int(departed),
                'attrition_rate': float(departed / total) if total > 0 else 0.0,
                'overtime_rate': float(avg_overtime_rate)
            }
        
        return result
    
    def _analyze_by_tenure(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate attrition rates by tenure bucket.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            dict: Attrition rates by tenure bucket.
        """
        if 'TenureBucket' not in df.columns or 'Attrition' not in df.columns:
            return {}
        
        result = {}
        buckets = df['TenureBucket'].unique()
        
        for bucket in buckets:
            if pd.isna(bucket):
                continue
            subset = df[df['TenureBucket'] == bucket]
            if len(subset) > 0:
                attrition_rate = (subset['Attrition'] == 'Yes').sum() / len(subset)
                result[str(bucket)] = float(attrition_rate)
        
        return result
    
    def _calculate_overtime_risk_ratio(self, df: pd.DataFrame) -> float:
        """
        Calculate the ratio of observed attrition rates for overtime vs non-overtime workers.
        
        This descriptive metric compares observed attrition rates; it does not establish causation.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            float: Risk ratio (overtime attrition rate / non-overtime attrition rate).
        """
        overtime_rates = self._analyze_by_overtime(df)
        
        no_rate = overtime_rates.get('No', 0.0)
        yes_rate = overtime_rates.get('Yes', 0.0)
        
        if no_rate == 0:
            return float('inf') if yes_rate > 0 else 1.0
        
        return yes_rate / no_rate
    
    def _get_demographic_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate demographic summary statistics.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            dict: Demographic summary including age, gender, and marital status breakdowns.
        """
        summary = {}
        
        # Age statistics
        if 'Age' in df.columns:
            summary['age'] = {
                'mean': float(df['Age'].mean()),
                'median': float(df['Age'].median()),
                'std': float(df['Age'].std()),
                'min': int(df['Age'].min()),
                'max': int(df['Age'].max())
            }
        
        # Gender distribution
        if 'Gender' in df.columns:
            summary['gender'] = df['Gender'].value_counts().to_dict()
        
        # Marital status distribution
        if 'MaritalStatus' in df.columns:
            summary['marital_status'] = df['MaritalStatus'].value_counts().to_dict()
        
        return summary
    
    def _calculate_financial_exposure(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate financial exposure from attrition.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            dict: Financial metrics including total replacement cost.
        """
        exposure = {}
        
        if 'ReplacementCost' in df.columns:
            # Total replacement cost for departed employees
            departed = df[df['Attrition'] == 'Yes']
            exposure['total_replacement_cost_departed'] = float(departed['ReplacementCost'].sum())
            
            # Average replacement cost per departure
            exposure['avg_replacement_cost_per_departure'] = float(departed['ReplacementCost'].mean())
            
            # Total exposure if all high-risk employees leave
            if 'IsOverTime' in df.columns:
                high_risk = df[df['IsOverTime'] == 1]
                exposure['potential_high_risk_exposure'] = float(high_risk['ReplacementCost'].sum())
        
        if 'AnnualIncome' in df.columns:
            exposure['total_annual_payroll'] = float(df['AnnualIncome'].sum())
            exposure['avg_annual_salary'] = float(df['AnnualIncome'].mean())
        
        return exposure
    
    def get_key_insights(self) -> List[str]:
        """
        Generate human-readable key insights from the analysis.
        
        Returns:
            list: List of insight strings.
        """
        insights = []
        
        if not self.analysis_results:
            return ["No analysis results available. Run analyze() first."]
        
        # Overall attrition insight
        overall_rate = self.analysis_results.get('overall_attrition_rate', 0)
        insights.append(f"Overall attrition rate is {overall_rate:.1%}")
        
        # Overtime insight
        overtime_ratio = self.analysis_results.get('overtime_risk_ratio', 1)
        overtime_rates = self.analysis_results.get('attrition_by_overtime', {})
        if overtime_ratio > 1:
            insights.append(
                f"Observed attrition among overtime employees was {overtime_ratio:.1f}x the rate "
                f"among non-overtime employees ({overtime_rates.get('Yes', 0):.1%} vs "
                f"{overtime_rates.get('No', 0):.1%}); this is an association, not a causal estimate."
            )
        
        # Financial insight
        financial = self.analysis_results.get('financial_exposure', {})
        if 'total_replacement_cost_departed' in financial:
            cost = financial['total_replacement_cost_departed']
            insights.append(f"Total replacement cost for departed employees: ${cost:,.0f}")
        
        return insights
