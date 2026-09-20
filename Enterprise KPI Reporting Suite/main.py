"""
Main Pipeline Runner for Enterprise KPI Reporting Suite
Orchestrates data loading, KPI calculation, and report generation.

Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import sys
from pathlib import Path
import logging
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import DataLoader
from src.anonymizer import DataAnonymizer
from src.kpi_engine import KPIAnalyticsEngine
from src.visualization import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(data_path: str, output_dir: str = 'reports'):
    """
    Execute the full KPI reporting pipeline.
    
    Args:
        data_path: Path to the input CSV file
        output_dir: Directory for output reports
    
    Returns:
        dict: Pipeline execution summary
    """
    logger.info("=" * 60)
    logger.info("Starting Enterprise KPI Reporting Pipeline")
    logger.info(f"Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA")
    logger.info("=" * 60)
    
    # Step 1: Load and validate data
    logger.info("\n[Step 1/5] Loading and validating data...")
    loader = DataLoader(data_path)
    df = loader.load()
    
    is_valid, missing_cols = loader.validate_schema()
    if not is_valid:
        logger.error(f"Schema validation failed. Missing: {missing_cols}")
        return {'success': False, 'error': 'Schema validation failed'}
    
    quality_report = loader.check_data_quality()
    logger.info(f"Data quality score: {quality_report['quality_score']}%")
    logger.info(f"Total records loaded: {quality_report['total_records']}")
    
    # Step 2: Anonymize sensitive data (for public sharing)
    logger.info("\n[Step 2/5] Anonymizing sensitive data...")
    anon = DataAnonymizer(df)
    anon.hash_employee_id('Employee_ID')
    anon_df = anon.get_anonymized_dataframe()
    anon_report = anon.get_anonymization_report()
    logger.info(f"Anonymized columns: {anon_report['anonymized_columns']}")
    
    # Step 3: Calculate KPIs
    logger.info("\n[Step 3/5] Calculating KPIs...")
    engine = KPIAnalyticsEngine(df)
    
    # Department KPIs
    dept_kpis = engine.calculate_department_kpis()
    logger.info("Department KPIs calculated")
    
    # Regional performance
    regional_kpis = engine.calculate_regional_performance()
    logger.info("Regional KPIs calculated")
    
    # Quarterly trends
    quarterly_trends = engine.calculate_quarterly_trends()
    logger.info("Quarterly trends calculated")
    
    # Health scores
    health_scores = engine.calculate_all_health_scores()
    logger.info(f"Department health scores: {health_scores}")
    
    # Underperforming areas
    underperforming = engine.identify_underperforming_areas(top_n=3)
    logger.info(f"Identified {len(underperforming)} underperforming areas")
    
    # Executive summary
    exec_summary = engine.generate_executive_summary()
    
    # Step 4: Generate visualizations
    logger.info("\n[Step 4/5] Generating visualizations...")
    report_gen = ReportGenerator(df, output_dir)
    
    # Create all charts
    report_gen.create_revenue_vs_cost_trend()
    report_gen.create_department_heatmap()
    report_gen.create_ltv_cac_distribution()
    report_gen.create_regional_performance_chart()
    report_gen.create_quarterly_trend_chart()
    report_gen.create_status_breakdown_pie()
    
    # Save figures
    saved_files = report_gen.save_all_figures(prefix='kpi_report')
    logger.info(f"Saved {len(saved_files)} visualization files")
    
    # Step 5: Export metrics JSON
    logger.info("\n[Step 5/5] Exporting executive metrics...")
    json_path = report_gen.generate_json_metrics(exec_summary)
    logger.info(f"Saved metrics JSON: {json_path}")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)
    
    pipeline_summary = {
        'success': True,
        'author': 'OGHENEOCHUKO EMMANUEL OGIDIAGBA',
        'records_processed': len(df),
        'total_revenue': exec_summary['data_scope']['total_revenue'],
        'total_cost': exec_summary['data_scope']['total_cost'],
        'below_target_count': exec_summary['performance']['records_below_target'],
        'department_health_scores': health_scores,
        'visualizations_generated': len(saved_files),
        'output_directory': output_dir
    }
    
    return pipeline_summary


def main():
    """Main entry point."""
    # Define paths
    base_dir = Path(__file__).parent
    data_path = base_dir / 'data' / 'KPI_Suite_Data.csv'
    output_dir = base_dir / 'reports'
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    result = run_pipeline(str(data_path), str(output_dir))
    
    # Print summary
    print("\n" + "=" * 60)
    print("ENTERPRISE KPI REPORTING SUITE - EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Author: {result.get('author', 'Unknown')}")
    print(f"Status: {'SUCCESS' if result.get('success') else 'FAILED'}")
    print(f"Records Processed: {result.get('records_processed', 0):,}")
    print(f"Total Revenue: ${result.get('total_revenue', 0)/1e9:.2f} billion")
    print(f"Below Target Records: {result.get('below_target_count', 0):,}")
    print(f"Visualizations Generated: {result.get('visualizations_generated', 0)}")
    print("=" * 60)
    
    # Print department health scores
    print("\nDepartment Health Scores:")
    for dept, score in result.get('department_health_scores', {}).items():
        status = "✓" if score >= 70 else "⚠" if score >= 50 else "✗"
        print(f"  {status} {dept}: {score}/100")
    
    print("\nReports saved to:", str(Path(__file__).parent / 'reports'))
    print("\nTo view the interactive dashboard, run:")
    print("  streamlit run dashboard.py")
    print("=" * 60)
    
    return result


if __name__ == '__main__':
    main()
