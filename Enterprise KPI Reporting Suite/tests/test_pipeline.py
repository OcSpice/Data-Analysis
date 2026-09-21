"""Tests for the Enterprise KPI Reporting Suite analytical pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.kpi_engine import KPIAnalyticsEngine
from src.visualization import ReportGenerator


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Record_ID": ["R1", "R2", "R3", "R4"],
        "Date": ["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15"],
        "Year": [2024, 2024, 2024, 2024],
        "Month": ["Jan", "Feb", "Mar", "Apr"],
        "Quarter": ["Q1", "Q1", "Q1", "Q2"],
        "Department": ["Finance", "Sales", "HR", "Operations"],
        "Region": ["North America", "Europe", "Asia Pacific", "Latin America"],
        "Product": ["Pro Plan"] * 4,
        "Channel": ["Direct Sales"] * 4,
        "Employee_ID": ["E1", "E2", "E3", "E4"],
        "Cost_Center": ["CC-1", "CC-2", "CC-3", "CC-4"],
        "Revenue": [10000, 15000, 8000, 12000],
        "Cost": [6000, 9000, 5000, 7000],
        "Gross_Margin": [4000, 6000, 3000, 5000],
        "Margin_Pct": [40.0, 40.0, 37.5, 41.6667],
        "Deals_Closed": [5, 8, 0, 3],
        "Leads_Generated": [20, 30, 15, 25],
        "Conv_Rate_Pct": [25.0, 26.6667, 0.0, 12.0],
        "Customer_Sat": [8.5, 9.0, 7.5, 8.0],
        "NPS": [70, 75, 60, 65],
        "Headcount": [150, 200, 100, 175],
        "Attrition_Pct": [10.0, 8.0, 15.0, 11.0],
        "Training_Hrs": [25, 30, 15, 20],
        "Productivity_Pct": [75.0, 80.0, 65.0, 72.0],
        "CAC": [500, 600, 400, 550],
        "LTV": [2000, 2500, 1500, 2200],
        "LTV_CAC_Ratio": [4.0, 4.1667, 3.75, 4.0],
        "MQL_Count": [100, 150, 80, 120],
        "SQL_Count": [50, 75, 40, 60],
        "CTR_Pct": [3.5, 4.0, 2.8, 3.2],
        "SLA_Met": [1, 1, 0, 1],
        "Ticket_Volume": [200, 250, 150, 180],
        "Resolution_Hrs": [10.5, 8.0, 15.0, 11.0],
        "Uptime_Pct": [99.8, 99.9, 99.2, 99.5],
        "Status": ["Above Target", "Above Target", "Below Target", "Above Target"],
    })


@pytest.fixture
def temp_csv_path(tmp_path, sample_df):
    path = tmp_path / "data.csv"
    sample_df.to_csv(path, index=False)
    return str(path)


class TestDataLoader:
    def test_load_and_schema(self, temp_csv_path):
        loader = DataLoader(temp_csv_path)
        df = loader.load()
        valid, missing = loader.validate_schema()
        assert len(df) == 4
        assert valid
        assert missing == []

    def test_quality_dimensions(self, temp_csv_path):
        loader = DataLoader(temp_csv_path)
        loader.load()
        report = loader.check_data_quality()
        assert report["dimensions"]["completeness_pct"] == 100.0
        assert report["dimensions"]["uniqueness_pct"] == 100.0
        assert report["dimensions"]["validity_pct"] == 100.0
        assert report["dimensions"]["consistency_pct"] == 100.0
        assert report["quality_score"] == 100.0

    def test_quality_catches_consistency_issue(self, sample_df, tmp_path):
        sample_df.loc[0, "Gross_Margin"] = 999
        path = tmp_path / "bad.csv"
        sample_df.to_csv(path, index=False)
        loader = DataLoader(str(path))
        loader.load()
        report = loader.check_data_quality()
        assert report["consistency_issues"]["margin_mismatch"] == 1


class TestKPIEngine:
    def test_department_kpis_use_weighted_margin(self, sample_df):
        engine = KPIAnalyticsEngine(sample_df)
        result = engine.calculate_department_kpis()
        assert "Record_Count" in result.columns
        assert np.isclose(result.loc["Finance", "Margin_Pct"], 40.0)

    def test_sales_conversion_is_derived_from_counts(self, sample_df):
        engine = KPIAnalyticsEngine(sample_df)
        result = engine.calculate_department_kpis()
        assert np.isclose(result.loc["Sales", "Conv_Rate_Pct"], 8 / 30 * 100, atol=0.01)

    def test_target_variance_is_explicit(self, sample_df):
        engine = KPIAnalyticsEngine(sample_df)
        variance = engine.get_department_target_variance("Finance")
        assert set(["Actual", "Target", "gap", "status", "relative_gap_pct"]).issubset(variance.columns)
        assert variance.loc[variance["KPI"] == "Margin_Pct", "status"].iloc[0] == "Meets Target"

    def test_lte_target_is_evaluated_in_correct_direction(self, sample_df):
        engine = KPIAnalyticsEngine(sample_df)
        variance = engine.get_department_target_variance("HR")
        attrition = variance[variance["KPI"] == "Attrition_Pct"].iloc[0]
        assert attrition["status"] == "Below Target"

    def test_exceptions_have_rule_based_priority(self, sample_df):
        engine = KPIAnalyticsEngine(sample_df)
        exceptions = engine.build_management_exceptions()
        assert not exceptions.empty
        assert set(exceptions["Priority"]).issubset({"High", "Medium", "Low"})

    def test_quarterly_trend_contains_period_variance(self, sample_df):
        engine = KPIAnalyticsEngine(sample_df)
        trends = engine.calculate_quarterly_trends()
        assert "Revenue_Change_Pct" in trends.columns
        assert "Margin_Change_Pp" in trends.columns
        assert len(trends) == 2

    def test_no_composite_health_score_api(self, sample_df):
        engine = KPIAnalyticsEngine(sample_df)
        assert not hasattr(engine, "calculate_department_health_score")

    def test_executive_summary_is_auditable(self, sample_df):
        summary = KPIAnalyticsEngine(sample_df).generate_executive_summary()
        assert "data_scope" in summary
        assert "performance" in summary
        assert "department_health_scores" not in summary
        assert "revenue_scope_label" not in summary["data_scope"]


class TestVisualization:
    def test_target_heatmap(self, sample_df):
        engine = KPIAnalyticsEngine(sample_df)
        report = ReportGenerator(sample_df, targets=engine.targets)
        fig = report.create_target_variance_heatmap(engine.get_department_target_variance())
        assert fig is not None
        assert len(fig.data) == 1

    def test_ltv_chart_uses_central_target(self, sample_df):
        engine = KPIAnalyticsEngine(sample_df)
        report = ReportGenerator(sample_df, targets=engine.targets)
        fig = report.create_ltv_cac_distribution()
        assert any(
            getattr(shape, "y0", None) == 3.0 for shape in (fig.layout.shapes or [])
        )

    def test_report_json(self, sample_df, tmp_path):
        report = ReportGenerator(sample_df, str(tmp_path))
        path = report.generate_json_metrics({"total_revenue": 100})
        assert path.endswith("executive_metrics.json")


class TestIntegrity:
    def test_margin_formula(self, sample_df):
        assert np.allclose(
            sample_df["Gross_Margin"],
            sample_df["Revenue"] - sample_df["Cost"],
        )

    def test_ltv_cac_formula(self, sample_df):
        assert np.allclose(
            sample_df["LTV_CAC_Ratio"],
            sample_df["LTV"] / sample_df["CAC"],
            rtol=0.01,
        )
