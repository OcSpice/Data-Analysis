"""
Main Pipeline for Supply Chain and Logistics Performance Tracker
Orchestrates the complete analytics workflow from data loading to report generation.
"""

import os
import sys
from typing import Dict, Any

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_quality import load_and_validate_data, DataAnonymizer
from generate_data import generate_dataset
from analytics_engine import LogisticsAnalyticsEngine
from visualization import VisualizationEngine
from report_generator import ReportGenerator


class SupplyChainPipeline:
    """Main pipeline orchestrating the supply chain analytics workflow."""
    
    AUTHOR = "OGHENEOCHUKO EMMANUEL OGIDIAGBA"
    PROJECT_NAME = "Supply Chain and Logistics Performance Tracker"
    PORTFOLIO_CATEGORY = "Data Analysis"
    
    def __init__(self, data_path: str, base_output_dir: str = None):
        self.data_path = data_path
        self.base_output_dir = base_output_dir or os.path.dirname(data_path)
        self.df = None
        self.validation_report = None
        self.analytics_results = None
        self.visualization_paths = {}
        self.report_paths = {}
        
    def run(self, generate_visualizations: bool = True, 
            generate_reports: bool = True,
            save_anonymized: bool = True) -> Dict[str, Any]:
        """
        Execute the complete analytics pipeline.
        
        Args:
            generate_visualizations: Whether to create visualization files
            generate_reports: Whether to generate text/JSON reports
            save_anonymized: Whether to save an anonymized version of the data
            
        Returns:
            Dictionary containing pipeline results and output paths
        """
        print("=" * 70)
        print(f"{self.PROJECT_NAME}")
        print(f"Author: {self.AUTHOR}")
        print(f"Portfolio Category: {self.PORTFOLIO_CATEGORY}")
        print("=" * 70)
        
        # Step 1: Load and validate data
        print("\n[1/5] Loading and validating data...")
        self.df, self.validation_report = load_and_validate_data(self.data_path)
        print(f"      Loaded {len(self.df):,} records")
        print(f"      Validation passed: {self.validation_report['validation_passed']}")
        
        if self.validation_report.get('issues', {}).get('missing_values'):
            print(f"      Missing values handled: {self.validation_report['issues']['missing_values']}")
        
        # Step 2: Run analytics engine
        print("\n[2/5] Running analytics engine...")
        engine = LogisticsAnalyticsEngine(self.df)
        self.analytics_results = engine.run_full_analysis()
        
        financial = self.analytics_results['savings_analysis']
        print(f"      Observed SLA penalties: ${financial['observed_sla_penalties']:,.2f}")
        print(f"      35% cold-chain exposure scenario: ${financial['total_scenario_exposure_base_35pct']:,.2f}")
        
        # Step 3: Generate visualizations
        if generate_visualizations:
            print("\n[3/5] Generating visualizations...")
            viz_engine = VisualizationEngine(
                self.df, 
                output_dir=os.path.join(self.base_output_dir, 'visualizations')
            )
            self.visualization_paths = viz_engine.create_all_visualizations()
            for viz_type, path in self.visualization_paths.items():
                print(f"      Created: {viz_type} -> {path}")
        
        # Step 4: Generate reports
        if generate_reports:
            print("\n[4/5] Generating reports...")
            report_gen = ReportGenerator(
                self.df,
                self.analytics_results,
                output_dir=os.path.join(self.base_output_dir, 'reports')
            )
            self.report_paths = report_gen.generate_all_reports()
            for report_type, path in self.report_paths.items():
                print(f"      Created: {report_type} -> {path}")
        
        # Step 5: Save anonymized dataset (optional)
        if save_anonymized:
            print("\n[5/5] Creating anonymized dataset...")
            anonymizer = DataAnonymizer()
            df_anonymized = anonymizer.create_anonymized_dataset(self.df)
            anon_path = os.path.join(self.base_output_dir, 'data', 'supply_chain_shipments_anonymized.csv')
            os.makedirs(os.path.dirname(anon_path), exist_ok=True)
            df_anonymized.to_csv(anon_path, index=False)
            print(f"      Saved anonymized data: {anon_path}")
        
        # Summary
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 70)
        print(f"\nKey Results:")
        print(f"  - Total Shipments Analyzed: {len(self.df):,}")
        print(f"  - On-Time Delivery Rate: {self.df['On_Time'].mean() * 100:.1f}%")
        print(f"  - Observed SLA Penalties: ${financial['observed_sla_penalties']:,.2f}")
        print(f"  - 35% Scenario Exposure: ${financial['total_scenario_exposure_base_35pct']:,.2f}")
        print(f"\nOutput Files:")
        for path in list(self.visualization_paths.values()) + list(self.report_paths.values()):
            print(f"  - {path}")
        
        return {
            'success': True,
            'records_processed': len(self.df),
            'observed_sla_penalties': financial['observed_sla_penalties'],
            'scenario_exposure_35pct': financial['total_scenario_exposure_base_35pct'],
            'visualization_paths': self.visualization_paths,
            'report_paths': self.report_paths,
            'analytics_results': self.analytics_results
        }


def main():
    """Main entry point for the pipeline."""
    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_path = os.path.join(base_dir, 'data', 'supply_chain_shipments.csv')
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data generation script has been run first.")
        sys.exit(1)
    
    # Run pipeline
    pipeline = SupplyChainPipeline(data_path, base_dir)
    results = pipeline.run(
        generate_visualizations=True,
        generate_reports=True,
        save_anonymized=True
    )
    
    return results


if __name__ == "__main__":
    main()
