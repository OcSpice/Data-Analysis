"""
HR Attrition Analysis and Retention Cost Modeling Pipeline
Data Analysis Portfolio Project

This module provides the main pipeline for analyzing employee attrition,
calculating replacement costs, and generating retention recommendations.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import DataLoader
from data_validator import DataValidator
from pii_masker import PIIMasker
from feature_engineer import FeatureEngineer
from exploratory_analysis import ExploratoryAnalysis
from attrition_model import AttritionModel
from business_impact import BusinessImpactAnalyzer
from report_generator import ReportGenerator


def main():
    """Execute the full attrition analysis pipeline."""
    
    # Define paths
    base_dir = Path(__file__).parent
    data_path = base_dir / "HR-Employee-Attrition-Dataset.csv"
    reports_dir = base_dir / "reports"
    
    print("=" * 60)
    print("HR ATTRITION ANALYSIS AND RETENTION COST MODELING")
    print("Data Analysis Portfolio Project")
    print("=" * 60)
    
    # Step 1: Load Data
    print("\n[1/7] Loading data...")
    loader = DataLoader(data_path)
    df = loader.load()
    print(f"Loaded {len(df)} employee records with {len(df.columns)} features")
    
    # Step 2: Validate Data Quality
    print("\n[2/7] Validating data quality...")
    validator = DataValidator()
    validation_report = validator.validate(df)
    print(f"Validation complete: {validation_report['status']}")
    if validation_report['missing_columns']:
        print(f"Missing columns: {validation_report['missing_columns']}")
    
    # Step 3: Apply PII Masking
    print("\n[3/7] Applying PII masking...")
    masker = PIIMasker()
    df_masked = masker.mask(df, columns=['EmployeeNumber'])
    print("PII masking applied to sensitive fields")
    
    # Step 4: Feature Engineering
    print("\n[4/7] Engineering features...")
    engineer = FeatureEngineer()
    df_enriched = engineer.create_features(df_masked)
    print(f"Created features: ReplacementCost, AnnualIncome, TenureBucket")
    
    # Step 5: Exploratory Analysis
    print("\n[5/7] Running exploratory analysis...")
    explorer = ExploratoryAnalysis()
    insights = explorer.analyze(df_enriched)
    
    print(f"\nKey Findings:")
    print(f"  - Overall Attrition Rate: {insights['overall_attrition_rate']:.1%}")
    print(f"  - Attrition Rate (Overtime=Yes): {insights['attrition_by_overtime']['Yes']:.1%}")
    print(f"  - Attrition Rate (Overtime=No): {insights['attrition_by_overtime']['No']:.1%}")
    print(f"  - Overtime workers quit at {insights['overtime_risk_ratio']:.1f}x the rate of non-overtime workers")
    
    # Step 6: Predictive Modeling with SHAP
    print("\n[6/7] Training attrition prediction model...")
    model = AttritionModel()
    model_results = model.train_and_explain(df_enriched)
    
    print(f"Model Accuracy: {model_results['accuracy']:.1%}")
    print(f"Model AUC-ROC: {model_results['auc_roc']:.3f}")
    print(f"\nTop 5 Drivers of Attrition (SHAP values):")
    for i, (feature, importance) in enumerate(model_results['top_features'].items(), 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # Step 7: Business Impact Analysis
    print("\n[7/7] Calculating business impact...")
    analyzer = BusinessImpactAnalyzer(replacement_cost_multiplier=1.5)
    impact = analyzer.calculate_impact(df_enriched, insights, model_results)
    
    print(f"\nBUSINESS IMPACT SUMMARY:")
    print(f"  - Total Annual Replacement Cost Exposure: ${impact['total_replacement_cost']:,.0f}")
    print(f"  - At-Risk Employees (High Probability): {impact['high_risk_employees']['count']}")
    print(f"  - Potential Savings with Retention Plan: ${impact['potential_savings']:,.0f}")
    
    # Generate Reports
    print("\n" + "=" * 60)
    print("GENERATING REPORTS")
    print("=" * 60)
    
    reporter = ReportGenerator(reports_dir)
    reporter.generate_all_reports(
        df_enriched=df_enriched,
        insights=insights,
        model_results=model_results,
        impact=impact
    )
    
    print("\nPipeline completed successfully!")
    print(f"Reports saved to: {reports_dir}")
    
    return {
        'insights': insights,
        'model_results': model_results,
        'impact': impact
    }


if __name__ == "__main__":
    results = main()
