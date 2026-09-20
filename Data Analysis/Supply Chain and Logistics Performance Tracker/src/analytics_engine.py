"""
Advanced Logistics Analytics Engine for Supply Chain Performance Tracker
Implements carrier scorecards, cold-chain risk analysis, and savings calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List


class CarrierScorecard:
    """Generates comprehensive carrier performance scorecards."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def calculate_metrics(self) -> pd.DataFrame:
        """Calculate key performance metrics for each carrier."""
        carrier_metrics = self.df.groupby('Carrier').agg({
            'Shipment_ID': 'count',
            'On_Time': 'mean',
            'Delay_Days': 'mean',
            'SLA_Penalty': 'sum',
            'Freight_Cost': 'sum',
            'Cargo_Value': 'sum',
            'Customs_Hold': 'sum',
            'Delivered': 'mean'
        }).reset_index()
        
        carrier_metrics.columns = [
            'Carrier', 'Total_Shipments', 'On_Time_Rate', 'Avg_Delay_Days',
            'Total_SLA_Penalty', 'Total_Freight_Cost', 'Total_Cargo_Value',
            'Total_Customs_Holds', 'Delivery_Success_Rate'
        ]
        
        # Calculate additional derived metrics
        carrier_metrics['Penalty_Rate'] = (
            carrier_metrics['Total_SLA_Penalty'] / carrier_metrics['Total_Freight_Cost'] * 100
        )
        
        # Composite score: weighted combination of metrics
        carrier_metrics['Performance_Score'] = (
            carrier_metrics['On_Time_Rate'] * 40 +
            (1 - carrier_metrics['Avg_Delay_Days'] / carrier_metrics['Avg_Delay_Days'].max()) * 30 +
            (1 - carrier_metrics['Penalty_Rate'] / carrier_metrics['Penalty_Rate'].max()) * 30
        )
        
        # Rank carriers
        carrier_metrics['Rank'] = carrier_metrics['Performance_Score'].rank(ascending=False).astype(int)
        
        return carrier_metrics.sort_values('Rank')
    
    def get_top_performers(self, n: int = 3) -> pd.DataFrame:
        """Return top N performing carriers."""
        metrics = self.calculate_metrics()
        return metrics.head(n)
    
    def get_underperformers(self, n: int = 3) -> pd.DataFrame:
        """Return bottom N performing carriers."""
        metrics = self.calculate_metrics()
        return metrics.tail(n).sort_values('Rank')


class ColdChainRiskAnalyzer:
    """Analyzes risks in cold-chain logistics for perishables and pharma."""
    
    COLD_CHAIN_TYPES = ['Perishables', 'Pharma']
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cold_chain_df = df[df['Cargo_Type'].isin(self.COLD_CHAIN_TYPES)].copy()
        
    def identify_high_risk_routes(self) -> pd.DataFrame:
        """Identify high-risk origin-destination pairs for cold-chain shipments."""
        if len(self.cold_chain_df) == 0:
            return pd.DataFrame()
            
        route_risk = self.cold_chain_df.groupby(['Origin', 'Destination']).agg({
            'Shipment_ID': 'count',
            'Delay_Days': 'mean',
            'On_Time': 'mean',
            'Customs_Hold': 'mean',
            'Cargo_Value': 'sum',
            'SLA_Penalty': 'sum'
        }).reset_index()
        
        route_risk.columns = [
            'Origin', 'Destination', 'Total_Shipments', 'Avg_Delay_Days',
            'On_Time_Rate', 'Customs_Hold_Rate', 'Total_Cargo_Value', 'Total_SLA_Penalty'
        ]
        
        # Calculate risk score
        route_risk['Risk_Score'] = (
            (1 - route_risk['On_Time_Rate']) * 40 +
            route_risk['Customs_Hold_Rate'] * 30 +
            (route_risk['Avg_Delay_Days'] / route_risk['Avg_Delay_Days'].max()) * 30
        )
        
        route_risk['Risk_Level'] = pd.cut(
            route_risk['Risk_Score'],
            bins=[0, 30, 60, 100],
            labels=['Low', 'Medium', 'High']
        )
        
        return route_risk.sort_values('Risk_Score', ascending=False)
    
    def analyze_by_mode(self) -> pd.DataFrame:
        """Analyze cold-chain performance by transportation mode."""
        if len(self.cold_chain_df) == 0:
            return pd.DataFrame()
            
        mode_analysis = self.cold_chain_df.groupby('Mode').agg({
            'Shipment_ID': 'count',
            'Delay_Days': 'mean',
            'On_Time': 'mean',
            'Customs_Hold': 'mean',
            'SLA_Penalty': 'sum'
        }).reset_index()
        
        mode_analysis.columns = [
            'Mode', 'Total_Shipments', 'Avg_Delay_Days', 'On_Time_Rate',
            'Customs_Hold_Rate', 'Total_SLA_Penalty'
        ]
        
        return mode_analysis.sort_values('Avg_Delay_Days', ascending=False)
    
    def get_delayed_shipments_value_at_risk(self) -> Dict[str, Any]:
        """Calculate value at risk for delayed cold-chain shipments."""
        delayed = self.cold_chain_df[self.cold_chain_df['Delay_Days'] > 0]
        
        if len(delayed) == 0:
            return {'total_value_at_risk': 0, 'shipment_count': 0}
        
        # Assume spoilage rate based on delay severity
        delayed = delayed.copy()
        delayed['Spoilage_Probability'] = np.clip(delayed['Delay_Days'] / 30, 0, 1)
        delayed['Value_At_Risk'] = delayed['Cargo_Value'] * delayed['Spoilage_Probability']
        
        return {
            'total_value_at_risk': delayed['Value_At_Risk'].sum(),
            'shipment_count': len(delayed),
            'avg_value_per_shipment': delayed['Cargo_Value'].mean(),
            'total_cargo_value': delayed['Cargo_Value'].sum()
        }


class SavingsCalculator:
    """Calculates cost savings from operational improvements."""
    
    # Configuration for savings calculations
    COLD_CHAIN_SPOILAGE_RATE = 0.35  # 35% of cargo value lost without intervention
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def calculate_recoverable_sla_penalties(self) -> float:
        """Calculate total recoverable SLA penalties from delayed shipments."""
        delayed = self.df[self.df['Delay_Days'] > 0]
        return delayed['SLA_Penalty'].sum()
    
    def calculate_cold_chain_loss_prevention(self) -> Dict[str, Any]:
        """
        Calculate prevented losses from fixing cold-chain routing problems.
        Identifies shipments where Delay_Days > 0 for Cargo_Type == 'Perishables'
        and calculates avoided spoilage cost.
        """
        perishables_delayed = self.df[
            (self.df['Cargo_Type'] == 'Perishables') & 
            (self.df['Delay_Days'] > 0)
        ].copy()
        
        if len(perishables_delayed) == 0:
            return {
                'prevented_loss': 0,
                'affected_shipments': 0,
                'total_cargo_value': 0
            }
        
        # Calculate potential loss without intervention
        perishables_delayed['Potential_Loss'] = (
            perishables_delayed['Cargo_Value'] * self.COLD_CHAIN_SPOILAGE_RATE
        )
        
        total_prevented = perishables_delayed['Potential_Loss'].sum()
        
        return {
            'prevented_loss': total_prevented,
            'affected_shipments': len(perishables_delayed),
            'total_cargo_value': perishables_delayed['Cargo_Value'].sum(),
            'avg_cargo_value': perishables_delayed['Cargo_Value'].mean()
        }
    
    def calculate_total_savings(self) -> Dict[str, Any]:
        """
        Calculate total savings combining SLA penalty recovery and cold-chain loss prevention.
        Target: $8.2 million total savings.
        """
        sla_recovery = self.calculate_recoverable_sla_penalties()
        cold_chain = self.calculate_cold_chain_loss_prevention()
        
        total_savings = sla_recovery + cold_chain['prevented_loss']
        
        return {
            'total_savings': total_savings,
            'sla_penalty_recovery': sla_recovery,
            'cold_chain_loss_prevention': cold_chain['prevented_loss'],
            'cold_chain_details': cold_chain,
            'breakdown': {
                'sla_recovery_percentage': (sla_recovery / total_savings * 100) if total_savings > 0 else 0,
                'cold_chain_percentage': (cold_chain['prevented_loss'] / total_savings * 100) if total_savings > 0 else 0
            }
        }


class DelayAnalyzer:
    """Analyzes delay patterns and root causes."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def get_monthly_delay_trends(self) -> pd.DataFrame:
        """Calculate monthly delay rates."""
        monthly = self.df.groupby(['Year', 'Month']).agg({
            'Shipment_ID': 'count',
            'Delay_Days': ['sum', 'mean'],
            'On_Time': 'mean'
        }).reset_index()
        
        monthly.columns = ['Year', 'Month', 'Total_Shipments', 'Total_Delay_Days', 
                          'Avg_Delay_Days', 'On_Time_Rate']
        monthly['Delay_Rate'] = 1 - monthly['On_Time_Rate']
        
        return monthly.sort_values(['Year', 'Month'])
    
    def analyze_delays_by_category(self, category: str) -> pd.DataFrame:
        """Analyze delays by a specific category (Carrier, Mode, Cargo_Type, etc.)."""
        if category not in self.df.columns:
            raise ValueError(f"Category '{category}' not found in data")
        
        analysis = self.df.groupby(category).agg({
            'Shipment_ID': 'count',
            'Delay_Days': ['sum', 'mean'],
            'On_Time': 'mean',
            'SLA_Penalty': 'sum'
        }).reset_index()
        
        analysis.columns = [category, 'Total_Shipments', 'Total_Delay_Days',
                           'Avg_Delay_Days', 'On_Time_Rate', 'Total_SLA_Penalty']
        analysis['Delay_Rate'] = 1 - analysis['On_Time_Rate']
        
        return analysis.sort_values('Avg_Delay_Days', ascending=False)
    
    def identify_delay_hotspots(self) -> Dict[str, Any]:
        """Identify the biggest contributors to delays."""
        # By carrier
        carrier_delays = self.analyze_delays_by_category('Carrier')
        top_delay_carrier = carrier_delays.iloc[0][carrier_delays.columns[0]]
        
        # By mode
        mode_delays = self.analyze_delays_by_category('Mode')
        top_delay_mode = mode_delays.iloc[0][mode_delays.columns[0]]
        
        # By cargo type
        cargo_delays = self.analyze_delays_by_category('Cargo_Type')
        top_delay_cargo = cargo_delays.iloc[0][cargo_delays.columns[0]]
        
        return {
            'highest_delay_carrier': top_delay_carrier,
            'highest_delay_mode': top_delay_mode,
            'highest_delay_cargo_type': top_delay_cargo,
            'overall_avg_delay': self.df['Delay_Days'].mean(),
            'overall_delay_rate': (self.df['Delay_Days'] > 0).mean()
        }


class LogisticsAnalyticsEngine:
    """Main analytics engine orchestrating all analysis components."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.carrier_scorecard = CarrierScorecard(df)
        self.cold_chain_analyzer = ColdChainRiskAnalyzer(df)
        self.savings_calculator = SavingsCalculator(df)
        self.delay_analyzer = DelayAnalyzer(df)
        
    def run_full_analysis(self) -> Dict[str, Any]:
        """Execute complete analytics pipeline."""
        results = {
            'carrier_scorecard': self.carrier_scorecard.calculate_metrics().to_dict('records'),
            'top_carriers': self.carrier_scorecard.get_top_performers().to_dict('records'),
            'high_risk_routes': self.cold_chain_analyzer.identify_high_risk_routes().to_dict('records'),
            'mode_analysis': self.cold_chain_analyzer.analyze_by_mode().to_dict('records'),
            'value_at_risk': self.cold_chain_analyzer.get_delayed_shipments_value_at_risk(),
            'savings_analysis': self.savings_calculator.calculate_total_savings(),
            'delay_trends': self.delay_analyzer.get_monthly_delay_trends().to_dict('records'),
            'delay_hotspots': self.delay_analyzer.identify_delay_hotspots()
        }
        
        return results
