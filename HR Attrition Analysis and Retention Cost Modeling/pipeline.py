"""
End-to-end HR attrition analysis pipeline.
"""

import sys
from pathlib import Path

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
    """Execute the complete HR attrition analysis."""
    base_dir = Path(__file__).parent
    data_path = base_dir / "HR-Employee-Attrition-Dataset.csv"
    reports_dir = base_dir / "reports"

    print("=" * 72)
    print("HR ATTRITION ANALYSIS AND RETENTION COST MODELING")
    print("=" * 72)

    print("\n[1/7] Loading data")
    df = DataLoader(data_path).load()
    print(f"Loaded {len(df):,} employee records and {len(df.columns)} source columns")

    print("\n[2/7] Validating data quality")
    validation = DataValidator().validate(df)
    print(f"Validation status: {validation['status']}")
    if validation["issues"]:
        for issue in validation["issues"]:
            print(f"  - {issue}")

    print("\n[3/7] Masking employee identifiers")
    df_masked = PIIMasker(salt="portfolio-hr-analysis").mask(
        df, columns=["EmployeeNumber"]
    )

    print("\n[4/7] Engineering analytical features")
    df_enriched = FeatureEngineer(replacement_cost_multiplier=1.5).create_features(
        df_masked
    )

    print("\n[5/7] Running workforce analysis")
    explorer = ExploratoryAnalysis()
    insights = explorer.analyze(df_enriched)

    print(f"Overall attrition: {insights['overall_attrition_rate']:.1%}")
    overtime = insights["attrition_by_overtime"]
    print(
        f"Observed attrition — overtime: {overtime.get('Yes', 0):.1%}; "
        f"non-overtime: {overtime.get('No', 0):.1%}"
    )
    print(f"Observed overtime rate ratio: {insights['overtime_risk_ratio']:.2f}x")

    print("\n[6/7] Training and explaining attrition models")
    model = AttritionModel(random_state=42)
    model_results = model.train_and_explain(df_enriched)

    for name, metrics in model_results["models"].items():
        print(
            f"{name}: ROC-AUC={metrics['roc_auc']:.3f}, "
            f"PR-AUC={metrics['pr_auc']:.3f}, "
            f"Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
        )
    print("Top model explanation features:")
    for feature, value in list(model_results["top_features"].items())[:5]:
        print(f"  - {feature}: {value:.5f}")

    print("\n[7/7] Calculating business impact")
    analyzer = BusinessImpactAnalyzer(
        replacement_cost_multiplier=1.5,
        high_risk_threshold=0.50,
    )
    impact = analyzer.calculate_impact(df_enriched, insights, model_results)

    print(
        f"Observed replacement-cost exposure: "
        f"{impact['total_replacement_cost']:,.0f} USD"
    )
    print("Illustrative retention scenarios:")
    for scenario in impact["retention_scenarios"]:
        print(
            f"  - {scenario['retention_effectiveness']:.0%}: "
            f"{scenario['avoided_cost_estimate']:,.0f} USD"
        )

    print("\nGenerating reports")
    ReportGenerator(reports_dir).generate_all_reports(
        df_enriched=df_enriched,
        insights=insights,
        model_results=model_results,
        impact=impact,
        validation_report=validation,
    )

    print("\nPipeline completed successfully.")
    return {
        "validation": validation,
        "insights": insights,
        "model_results": model_results,
        "impact": impact,
    }


if __name__ == "__main__":
    main()
