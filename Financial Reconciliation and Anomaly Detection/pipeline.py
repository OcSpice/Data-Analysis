"""
Financial Reconciliation and Anomaly Detection Pipeline
========================================================

This module serves as the main entry point for the financial reconciliation
and anomaly detection system. It orchestrates data loading, validation,
PII masking, anomaly detection, and reporting.

Portfolio Category: Data Analysis (enhanced with Machine Learning techniques)
Author: Oghenochuko Emmanuel Ogidiagba
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data_loader import DataLoader
from data_quality_engine import DataQualityEngine
from pii_masker import PIIMasker
from anomaly_detector import AnomalyDetector
from report_generator import ReportGenerator


def main():
    """
    Main execution pipeline for financial reconciliation and anomaly detection.
    
    This function orchestrates the complete workflow:
    1. Load raw transaction data
    2. Validate data quality and schema
    3. Apply PII masking for compliance
    4. Detect anomalies using statistical and ML methods
    5. Generate comprehensive reports and visualizations
    """
    # Define paths
    base_dir = Path(__file__).parent
    data_path = base_dir / "Financial_Reconciliation_Data.csv"
    reports_dir = base_dir / "reports"
    
    # Ensure reports directory exists
    reports_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("FINANCIAL RECONCILIATION AND ANOMALY DETECTION PIPELINE")
    print("=" * 60)
    print(f"Portfolio Category: Data Analysis")
    print(f"Data Source: {data_path}")
    print()
    
    # Step 1: Load Data
    print("[Step 1] Loading transaction data...")
    loader = DataLoader(data_path)
    df = loader.load_data()
    print(f"      Loaded {len(df):,} transactions")
    print()
    
    # Step 2: Data Quality Validation
    print("[Step 2] Running data quality validation...")
    dqe = DataQualityEngine(df)
    quality_report = dqe.run_full_validation()
    print(f"      Schema validation: {'PASSED' if quality_report['schema_valid'] else 'FAILED'}")
    print(f"      Missing values found: {quality_report['total_missing']}")
    print(f"      Duplicates identified: {quality_report['duplicate_count']}")
    print(f"      Data quality score: {quality_report['quality_score']:.2%}")
    print()
    
    # Get cleaned data
    df_clean = dqe.get_cleaned_data()
    
    # Step 3: PII Masking
    print("[Step 3] Applying PII masking for compliance...")
    masker = PIIMasker(df_clean)
    df_masked = masker.mask_all_pii()
    print(f"      Masked columns: {masker.masked_columns}")
    print()
    
    # Step 4: Anomaly Detection
    print("[Step 4] Running anomaly detection engine...")
    detector = AnomalyDetector(df_masked)
    
    # Run statistical analysis (Z-score)
    anomalies_zscore = detector.detect_zscore_anomalies(threshold=3.0)
    print(f"      Z-score anomalies detected: {len(anomalies_zscore)}")
    
    # Run Isolation Forest
    anomalies_if = detector.detect_isolation_forest_anomalies(contamination=0.05)
    print(f"      Isolation Forest anomalies detected: {len(anomalies_if)}")
    
    # Combine and tier anomalies
    anomaly_summary = detector.create_audit_priority_tiering()
    print(f"      High priority audits: {anomaly_summary.get('high_priority', 0)}")
    print(f"      Medium priority audits: {anomaly_summary.get('medium_priority', 0)}")
    print(f"      Low priority audits: {anomaly_summary.get('low_priority', 0)}")
    print()
    
    # Step 5: Generate Reports
    print("[Step 5] Generating reports and visualizations...")
    
    # Get the DataFrame with anomaly flags from detector (already has flags from previous calls)
    df_with_anomalies = detector.df.copy()
    reporter = ReportGenerator(df_with_anomalies, detector, reports_dir)
    
    # Generate all reports
    reporter.generate_summary_report()
    reporter.generate_department_risk_analysis()
    reporter.generate_source_system_reliability()
    reporter.generate_temporal_trends()
    reporter.generate_financial_exposure_report()
    
    print(f"      Reports saved to: {reports_dir}")
    print()
    
    print("=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    return {
        "data_quality": quality_report,
        "anomaly_summary": anomaly_summary,
        "reports_path": str(reports_dir)
    }


if __name__ == "__main__":
    results = main()
