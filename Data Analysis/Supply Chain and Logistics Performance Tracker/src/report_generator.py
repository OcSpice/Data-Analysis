"""
Report Generator for Supply Chain Analytics Pipeline
Generates JSON metrics, text reports, and executive summaries with author attribution.
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
import os


class ReportGenerator:
    """Generates comprehensive reports from supply chain analytics."""
    
    # Author metadata - persists across all pipeline runs
    AUTHOR = "OGHENEOCHUKO EMMANUEL OGIDIAGBA"
    PROJECT_NAME = "Supply Chain and Logistics Performance Tracker"
    PORTFOLIO_CATEGORY = "Data Analysis"
    VERSION = "1.0.0"
    
    def __init__(self, df: pd.DataFrame, analytics_results: Dict[str, Any], 
                 output_dir: str = 'reports'):
        self.df = df.copy()
        self.analytics_results = analytics_results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
    def generate_key_metrics(self) -> Dict[str, Any]:
        """
        Generate key performance metrics in JSON format.
        Includes author metadata that persists across pipeline re-runs.
        """
        savings = self.analytics_results.get('savings_analysis', {})
        delay_hotspots = self.analytics_results.get('delay_hotspots', {})
        
        # Calculate additional metrics
        total_shipments = len(self.df)
        on_time_rate = self.df['On_Time'].mean() * 100
        avg_delay = self.df[self.df['Delay_Days'] > 0]['Delay_Days'].mean()
        total_freight_cost = self.df['Freight_Cost'].sum()
        total_cargo_value = self.df['Cargo_Value'].sum()
        
        metrics = {
            'metadata': {
                'author': self.AUTHOR,
                'project_name': self.PROJECT_NAME,
                'portfolio_category': self.PORTFOLIO_CATEGORY,
                'version': self.VERSION,
                'generated_at': self.generated_at,
                'data_period': f"{self.df['Date'].min()} to {self.df['Date'].max()}"
            },
            'operational_metrics': {
                'total_shipments': total_shipments,
                'on_time_delivery_rate_pct': round(on_time_rate, 2),
                'average_delay_days': round(avg_delay, 2) if pd.notna(avg_delay) else 0,
                'total_delayed_shipments': int((self.df['Delay_Days'] > 0).sum()),
                'customs_hold_rate_pct': round(self.df['Customs_Hold'].mean() * 100, 2)
            },
            'financial_metrics': {
                'total_freight_cost_usd': round(total_freight_cost, 2),
                'total_cargo_value_usd': round(total_cargo_value, 2),
                'total_sla_penalties_usd': round(self.df['SLA_Penalty'].sum(), 2),
                'avg_freight_cost_per_shipment': round(total_freight_cost / total_shipments, 2)
            },
            'savings_analysis': {
                'total_identified_savings_usd': round(savings.get('total_savings', 0), 2),
                'sla_penalty_recovery_usd': round(savings.get('sla_penalty_recovery', 0), 2),
                'cold_chain_loss_prevention_usd': round(savings.get('cold_chain_loss_prevention', 0), 2),
                'savings_breakdown': {
                    'sla_recovery_percentage': round(savings.get('breakdown', {}).get('sla_recovery_percentage', 0), 2),
                    'cold_chain_percentage': round(savings.get('breakdown', {}).get('cold_chain_percentage', 0), 2)
                }
            },
            'delay_insights': {
                'highest_delay_carrier': delay_hotspots.get('highest_delay_carrier', 'N/A'),
                'highest_delay_mode': delay_hotspots.get('highest_delay_mode', 'N/A'),
                'highest_delay_cargo_type': delay_hotspots.get('highest_delay_cargo_type', 'N/A'),
                'overall_delay_rate_pct': round(delay_hotspots.get('overall_delay_rate', 0) * 100, 2)
            }
        }
        
        return metrics
    
    def save_metrics_json(self, filename: str = 'key_metrics.json') -> str:
        """Save key metrics to JSON file."""
        metrics = self.generate_key_metrics()
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return output_path
    
    def generate_executive_summary(self) -> str:
        """Generate an executive summary text report."""
        metrics = self.generate_key_metrics()
        savings = metrics['savings_analysis']
        ops = metrics['operational_metrics']
        
        summary = f"""
================================================================================
EXECUTIVE SUMMARY: {self.PROJECT_NAME}
Portfolio Category: {self.PORTFOLIO_CATEGORY}
Generated: {self.generated_at}
Author: {self.AUTHOR}
================================================================================

OVERVIEW
--------
This analysis examines {ops['total_shipments']:,} shipment records to identify operational
inefficiencies, carrier performance gaps, and cost-savings opportunities in the
supply chain logistics network.

KEY OPERATIONAL METRICS
-----------------------
- Total Shipments Analyzed: {ops['total_shipments']:,}
- On-Time Delivery Rate: {ops['on_time_delivery_rate_pct']:.1f}%
- Average Delay (for delayed shipments): {ops['average_delay_days']:.1f} days
- Total Delayed Shipments: {ops['total_delayed_shipments']:,}
- Customs Hold Rate: {ops['customs_hold_rate_pct']:.1f}%

FINANCIAL IMPACT
----------------
- Total Freight Costs: ${metrics['financial_metrics']['total_freight_cost_usd']:,.2f}
- Total Cargo Value: ${metrics['financial_metrics']['total_cargo_value_usd']:,.2f}
- Total SLA Penalties Incurred: ${metrics['financial_metrics']['total_sla_penalties_usd']:,.2f}

IDENTIFIED SAVINGS OPPORTUNITY: ${savings['total_identified_savings_usd']:,.2f}
---------------------------------------------------------------------------

SAVINGS BREAKDOWN
-----------------
1. SLA Penalty Recovery: ${savings['sla_penalty_recovery_usd']:,.2f}
   - Recoverable penalties from carriers for delayed shipments
   - Represents {savings['savings_breakdown']['sla_recovery_percentage']:.1f}% of total savings

2. Cold-Chain Loss Prevention: ${savings['cold_chain_loss_prevention_usd']:,.2f}
   - Prevented spoilage losses for perishables shipments
   - Identified high-risk routes and implemented corrective actions
   - Represents {savings['savings_breakdown']['cold_chain_percentage']:.1f}% of total savings

DELAY ROOT CAUSE ANALYSIS
-------------------------
- Highest Delay Carrier: {metrics['delay_insights']['highest_delay_carrier']}
- Highest Delay Mode: {metrics['delay_insights']['highest_delay_mode']}
- Highest Delay Cargo Type: {metrics['delay_insights']['highest_delay_cargo_type']}
- Overall Delay Rate: {metrics['delay_insights']['overall_delay_rate_pct']:.1f}%

RECOMMENDATIONS
---------------
1. Renegotiate SLA terms with underperforming carriers to increase penalty recovery
2. Implement proactive cold-chain monitoring for perishables on high-risk routes
3. Optimize routing for cargo types with highest delay rates
4. Enhance customs documentation processes to reduce hold rates

================================================================================
Report generated by {self.PROJECT_NAME} v{self.VERSION}
Author: {self.AUTHOR}
Portfolio: {self.PORTFOLIO_CATEGORY} Track
================================================================================
"""
        return summary
    
    def save_executive_summary(self, filename: str = 'executive_summary.txt') -> str:
        """Save executive summary to text file."""
        summary = self.generate_executive_summary()
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w') as f:
            f.write(summary)
        
        return output_path
    
    def generate_carrier_report(self) -> str:
        """Generate detailed carrier performance report."""
        carrier_data = self.analytics_results.get('carrier_scorecard', [])
        
        if not carrier_data:
            return "No carrier data available."
        
        report = f"""
================================================================================
CARRIER PERFORMANCE SCORECARD
{self.PROJECT_NAME}
Author: {self.AUTHOR}
Generated: {self.generated_at}
================================================================================

CARRIER RANKINGS BY PERFORMANCE SCORE
-------------------------------------
"""
        
        for carrier in carrier_data:
            report += f"""
Carrier: {carrier.get('Carrier', 'N/A')}
  Rank: #{carrier.get('Rank', 'N/A')}
  Total Shipments: {carrier.get('Total_Shipments', 0):,}
  On-Time Rate: {carrier.get('On_Time_Rate', 0) * 100:.1f}%
  Average Delay: {carrier.get('Avg_Delay_Days', 0):.1f} days
  Total SLA Penalties: ${carrier.get('Total_SLA_Penalty', 0):,.2f}
  Performance Score: {carrier.get('Performance_Score', 0):.1f}
"""
        
        report += f"""
================================================================================
Author Attribution: {self.AUTHOR}
Portfolio Category: {self.PORTFOLIO_CATEGORY}
================================================================================
"""
        return report
    
    def save_carrier_report(self, filename: str = 'carrier_scorecard.txt') -> str:
        """Save carrier report to text file."""
        report = self.generate_carrier_report()
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        return output_path
    
    def generate_all_reports(self) -> Dict[str, str]:
        """Generate and save all standard reports."""
        results = {}
        
        results['metrics_json'] = self.save_metrics_json()
        results['executive_summary'] = self.save_executive_summary()
        results['carrier_report'] = self.save_carrier_report()
        
        return results
