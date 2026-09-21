"""Run the Enterprise KPI Reporting Suite pipeline."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.anonymizer import DataAnonymizer
from src.kpi_engine import KPIAnalyticsEngine
from src.visualization import ReportGenerator

AUTHOR = "OGHENEOCHUKO EMMANUEL OGIDIAGBA"

def run_pipeline(data_path: str, output_dir: str = "reports") -> dict:
    loader = DataLoader(data_path)
    df = loader.load()
    schema_valid, missing = loader.validate_schema()
    if not schema_valid:
        return {"success": False, "error": f"Schema validation failed: {missing}"}

    quality = loader.check_data_quality()
    anonymizer = DataAnonymizer(df)
    anonymizer.hash_employee_id("Employee_ID")

    engine = KPIAnalyticsEngine(df)
    department_kpis = engine.calculate_department_kpis()
    regional_kpis = engine.calculate_regional_performance()
    quarterly = engine.calculate_quarterly_trends()
    target_variance = engine.get_department_target_variance()
    exceptions = engine.build_management_exceptions()
    executive = engine.generate_executive_summary()

    report = ReportGenerator(df, output_dir, targets=engine.targets)
    report.create_revenue_vs_cost_trend()
    report.create_target_variance_heatmap(target_variance)
    report.create_ltv_cac_distribution()
    report.create_regional_performance_chart()
    report.create_quarterly_trend_chart()
    report.create_status_breakdown_pie()
    report.create_exception_priority_chart(exceptions)
    saved_files = report.save_all_figures()
    json_path = report.generate_json_metrics({
        **executive,
        "data_quality": quality,
        "target_variance": target_variance.to_dict("records"),
        "management_exceptions": exceptions.to_dict("records"),
    })

    return {
        "success": True,
        "author": AUTHOR,
        "records_processed": len(df),
        "total_revenue": executive["data_scope"]["total_revenue"],
        "total_cost": executive["data_scope"]["total_cost"],
        "quality_score": quality["quality_score"],
        "management_exception_count": len(exceptions),
        "visualizations_generated": len(saved_files),
        "report_json": json_path,
        "department_count": len(department_kpis),
        "region_count": len(regional_kpis),
        "quarter_count": len(quarterly),
    }

def main() -> dict:
    base = Path(__file__).parent
    result = run_pipeline(str(base / "data" / "KPI_Suite_Data.csv"), str(base / "reports"))
    print("=" * 64)
    print("ENTERPRISE KPI REPORTING SUITE")
    print("=" * 64)
    print(f"Author: {result.get('author', 'Unknown')}")
    print(f"Status: {'SUCCESS' if result.get('success') else 'FAILED'}")
    print(f"Records processed: {result.get('records_processed', 0):,}")
    print(f"Reported revenue: {result.get('total_revenue', 0):,.2f}")
    print(f"Data quality score: {result.get('quality_score', 0):.2f}%")
    print(f"Management exceptions: {result.get('management_exception_count', 0):,}")
    print(f"Visualizations generated: {result.get('visualizations_generated', 0)}")
    print("=" * 64)
    return result

if __name__ == "__main__":
    main()
