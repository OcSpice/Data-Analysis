"""
KPI Analytics Engine for Enterprise KPI Reporting Suite
Calculates 12 core targets across Finance, Sales, HR, and Operations.

Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KPIAnalyticsEngine:
    """
    Advanced KPI calculation engine for enterprise performance tracking.
    
    Calculates 12 core targets across four departments:
    - Finance: Margin_Pct, Revenue vs Cost
    - Sales: LTV_CAC_Ratio, Conv_Rate_Pct, Deals_Closed
    - HR: Attrition_Pct, Productivity_Pct, Training_Hrs
    - Operations: SLA_Met, Uptime_Pct, Resolution_Hrs
    
    Attributes:
        df (pd.DataFrame): The input dataframe with KPI data
        targets (Dict): Dictionary of target thresholds for each metric
        department_health_scores (Dict): Calculated health scores by department
    """
    
    # Class-level constant for author metadata
    AUTHOR = "OGHENEOCHUKO EMMANUEL OGIDIAGBA"
    
    # Core business targets by department
    DEPARTMENT_TARGETS = {
        'Finance': {
            'Margin_Pct': {'target': 40.0, 'direction': 'gte'},  # >= 40%
            'Revenue_vs_Cost': {'target': 1.0, 'direction': 'gte'}  # Revenue >= Cost
        },
        'Sales': {
            'LTV_CAC_Ratio': {'target': 3.0, 'direction': 'gte'},  # >= 3:1
            'Conv_Rate_Pct': {'target': 15.0, 'direction': 'gte'},  # >= 15%
            'Deals_Closed': {'target': 5, 'direction': 'gte'}  # >= 5 deals
        },
        'HR': {
            'Attrition_Pct': {'target': 12.0, 'direction': 'lte'},  # <= 12%
            'Productivity_Pct': {'target': 70.0, 'direction': 'gte'},  # >= 70%
            'Training_Hrs': {'target': 20.0, 'direction': 'gte'}  # >= 20 hours
        },
        'Operations': {
            'SLA_Met': {'target': 1, 'direction': 'gte'},  # SLA met flag
            'Uptime_Pct': {'target': 99.5, 'direction': 'gte'},  # >= 99.5%
            'Resolution_Hrs': {'target': 12.0, 'direction': 'lte'}  # <= 12 hours
        }
    }
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the KPI Analytics Engine.
        
        Args:
            df: DataFrame containing enterprise KPI data
        """
        self.df = df.copy()
        self.targets = self.DEPARTMENT_TARGETS
        self.department_health_scores: Dict[str, float] = {}
        self.underperforming_areas: List[Dict] = []
    
    def calculate_department_kpis(self) -> pd.DataFrame:
        """
        Calculate aggregated KPIs by department.
        
        Returns:
            pd.DataFrame: Aggregated KPIs by department
        """
        kpi_summary = self.df.groupby('Department').agg({
            'Revenue': 'sum',
            'Cost': 'sum',
            'Gross_Margin': 'sum',
            'Margin_Pct': 'mean',
            'Deals_Closed': 'sum',
            'Leads_Generated': 'sum',
            'Conv_Rate_Pct': 'mean',
            'Customer_Sat': 'mean',
            'NPS': 'mean',
            'Headcount': 'mean',
            'Attrition_Pct': 'mean',
            'Training_Hrs': 'mean',
            'Productivity_Pct': 'mean',
            'CAC': 'mean',
            'LTV': 'mean',
            'LTV_CAC_Ratio': 'mean',
            'SLA_Met': 'mean',
            'Ticket_Volume': 'sum',
            'Resolution_Hrs': 'mean',
            'Uptime_Pct': 'mean'
        }).round(2)
        
        # Calculate derived metrics
        kpi_summary['Revenue_vs_Cost'] = (
            kpi_summary['Revenue'] / kpi_summary['Cost'].replace(0, np.nan)
        ).round(2)
        
        logger.info("Calculated department-level KPIs")
        
        return kpi_summary
    
    def calculate_regional_performance(self) -> pd.DataFrame:
        """
        Calculate KPIs aggregated by region.
        
        Returns:
            pd.DataFrame: Aggregated KPIs by region
        """
        regional_kpis = self.df.groupby('Region').agg({
            'Revenue': 'sum',
            'Cost': 'sum',
            'Gross_Margin': 'sum',
            'Margin_Pct': 'mean',
            'Deals_Closed': 'sum',
            'Attrition_Pct': 'mean',
            'Productivity_Pct': 'mean',
            'LTV_CAC_Ratio': 'mean',
            'SLA_Met': 'mean',
            'Uptime_Pct': 'mean'
        }).round(2)
        
        regional_kpis['Revenue_vs_Cost'] = (
            regional_kpis['Revenue'] / regional_kpis['Cost'].replace(0, np.nan)
        ).round(2)
        
        logger.info("Calculated regional KPIs")
        
        return regional_kpis
    
    def calculate_quarterly_trends(self) -> pd.DataFrame:
        """
        Calculate KPI trends by quarter.
        
        Returns:
            pd.DataFrame: Quarterly aggregated KPIs
        """
        quarterly = self.df.groupby(['Year', 'Quarter']).agg({
            'Revenue': 'sum',
            'Cost': 'sum',
            'Gross_Margin': 'sum',
            'Margin_Pct': 'mean',
            'Deals_Closed': 'sum',
            'Attrition_Pct': 'mean',
            'Productivity_Pct': 'mean',
            'LTV_CAC_Ratio': 'mean',
            'SLA_Met': 'mean',
            'Uptime_Pct': 'mean'
        }).reset_index().round(2)
        
        logger.info("Calculated quarterly trends")
        
        return quarterly
    
    def calculate_department_health_score(self, department: str) -> float:
        """
        Calculate a health score (0-100) for a specific department.
        
        Args:
            department: Name of the department
            
        Returns:
            float: Health score from 0 to 100
        """
        if department not in self.df['Department'].unique():
            logger.warning(f"Department '{department}' not found")
            return 0.0
        
        dept_data = self.df[self.df['Department'] == department]
        targets = self.targets.get(department, {})
        
        if not targets:
            return 50.0  # Default score if no targets defined
        
        scores = []
        
        for metric, target_info in targets.items():
            target_value = target_info['target']
            direction = target_info['direction']
            
            if metric == 'Revenue_vs_Cost':
                actual = (dept_data['Revenue'].sum() / dept_data['Cost'].sum())
            elif metric in dept_data.columns:
                actual = dept_data[metric].mean()
            else:
                continue
            
            # Calculate score based on direction
            if direction == 'gte':  # Higher is better
                if actual >= target_value:
                    score = 100
                else:
                    score = max(0, (actual / target_value) * 100)
            else:  # lte - Lower is better
                if actual <= target_value:
                    score = 100
                else:
                    score = max(0, (target_value / actual) * 100)
            
            scores.append(score)
        
        health_score = np.mean(scores) if scores else 50.0
        self.department_health_scores[department] = round(health_score, 2)
        
        logger.info(f"Health score for {department}: {health_score:.2f}")
        
        return round(health_score, 2)
    
    def calculate_all_health_scores(self) -> Dict[str, float]:
        """
        Calculate health scores for all departments.
        
        Returns:
            Dict[str, float]: Department names mapped to health scores
        """
        for department in self.df['Department'].unique():
            self.calculate_department_health_score(department)
        
        return self.department_health_scores
    
    def identify_underperforming_areas(self, top_n: int = 3) -> List[Dict]:
        """
        Identify top N underperforming cost centers or regions.
        
        Args:
            top_n: Number of underperforming areas to identify
            
        Returns:
            List[Dict]: List of underperforming areas with gap analysis
        """
        underperforming = []
        
        # Analyze by Cost Center
        cost_center_perf = self.df.groupby('Cost_Center').agg({
            'Revenue': 'sum',
            'Cost': 'sum',
            'Margin_Pct': 'mean',
            'Status': lambda x: (x == 'Below Target').sum() / len(x) * 100
        }).reset_index()
        
        cost_center_perf['Below_Target_Rate'] = cost_center_perf['Status'].round(2)
        cost_center_perf['Revenue_Gap'] = (
            cost_center_perf['Revenue'] - cost_center_perf['Cost']
        )
        
        # Find cost centers with high below-target rates
        worst_cost_centers = cost_center_perf.nlargest(top_n, 'Below_Target_Rate')
        
        for _, row in worst_cost_centers.iterrows():
            underperforming.append({
                'type': 'Cost_Center',
                'identifier': row['Cost_Center'],
                'below_target_rate': row['Below_Target_Rate'],
                'revenue': row['Revenue'],
                'cost': row['Cost'],
                'revenue_gap': row['Revenue_Gap'],
                'margin_pct': row['Margin_Pct']
            })
        
        # Analyze by Region
        region_perf = self.df.groupby('Region').agg({
            'Revenue': 'sum',
            'Cost': 'sum',
            'Margin_Pct': 'mean',
            'Status': lambda x: (x == 'Below Target').sum() / len(x) * 100
        }).reset_index()
        
        region_perf['Below_Target_Rate'] = region_perf['Status'].round(2)
        region_perf['Revenue_Gap'] = region_perf['Revenue'] - region_perf['Cost']
        
        worst_regions = region_perf.nlargest(top_n, 'Below_Target_Rate')
        
        for _, row in worst_regions.iterrows():
            underperforming.append({
                'type': 'Region',
                'identifier': row['Region'],
                'below_target_rate': row['Below_Target_Rate'],
                'revenue': row['Revenue'],
                'cost': row['Cost'],
                'revenue_gap': row['Revenue_Gap'],
                'margin_pct': row['Margin_Pct']
            })
        
        # Sort by below target rate
        underperforming.sort(key=lambda x: x['below_target_rate'], reverse=True)
        self.underperforming_areas = underperforming[:top_n]
        
        logger.info(f"Identified {len(self.underperforming_areas)} underperforming areas")
        
        return self.underperforming_areas
    
    def get_below_target_records(self) -> pd.DataFrame:
        """
        Filter and return records flagged as 'Below Target'.
        
        Returns:
            pd.DataFrame: Records with Below Target status
        """
        below_target = self.df[self.df['Status'] == 'Below Target'].copy()
        logger.info(f"Found {len(below_target)} records below target")
        
        return below_target
    
    def generate_executive_summary(self) -> Dict:
        """
        Generate an executive summary of enterprise performance.
        
        Returns:
            Dict: Executive summary with key metrics and insights
        """
        total_revenue = self.df['Revenue'].sum()
        total_cost = self.df['Cost'].sum()
        total_margin = self.df['Gross_Margin'].sum()
        overall_margin_pct = (total_margin / total_revenue * 100) if total_revenue > 0 else 0
        
        below_target_count = (self.df['Status'] == 'Below Target').sum()
        below_target_pct = (below_target_count / len(self.df) * 100)
        
        # Calculate all health scores
        self.calculate_all_health_scores()
        
        # Identify underperforming areas
        self.identify_underperforming_areas(top_n=3)
        
        summary = {
            'author': self.AUTHOR,
            'generated_at': datetime.now().isoformat(),
            'data_scope': {
                'total_records': len(self.df),
                'total_revenue': round(total_revenue, 2),
                'total_cost': round(total_cost, 2),
                'total_margin': round(total_margin, 2),
                'overall_margin_pct': round(overall_margin_pct, 2),
                'revenue_scope_label': '$1.4 billion revenue scope'
            },
            'performance': {
                'records_below_target': int(below_target_count),
                'below_target_percentage': round(below_target_pct, 2),
                'records_above_target': int(len(self.df) - below_target_count)
            },
            'department_health_scores': self.department_health_scores,
            'underperforming_areas': self.underperforming_areas,
            'departments_analyzed': list(self.df['Department'].unique()),
            'regions_analyzed': list(self.df['Region'].unique())
        }
        
        logger.info("Generated executive summary")
        
        return summary
    
    def get_kpi_targets_reference(self) -> Dict:
        """
        Return the reference targets used for KPI evaluation.
        
        Returns:
            Dict: Dictionary of all KPI targets by department
        """
        return {
            'author': self.AUTHOR,
            'targets': self.targets
        }
