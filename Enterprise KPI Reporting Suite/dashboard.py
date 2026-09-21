"""Streamlit dashboard for the Enterprise KPI Reporting Suite."""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.kpi_engine import KPIAnalyticsEngine
from src.visualization import ReportGenerator
from src.anonymizer import DataAnonymizer

AUTHOR = "OGHENEOCHUKO EMMANUEL OGIDIAGBA"

st.set_page_config(page_title="Enterprise KPI Reporting Suite", page_icon="📊", layout="wide")

@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    loader = DataLoader(path)
    df = loader.load()
    valid, missing = loader.validate_schema()
    if not valid:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def show_target_table(engine: KPIAnalyticsEngine, department: str | None = None):
    variance = engine.get_department_target_variance(department)
    if variance.empty:
        st.info("No target configuration is available for this selection.")
        return
    display = variance[
        ["Department", "KPI", "Actual", "Target", "Direction", "Unit", "gap",
         "relative_gap_pct", "status"]
    ].rename(columns={
        "gap": "Gap",
        "relative_gap_pct": "Gap vs Target (%)",
        "status": "Status",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

def main():
    data_path = Path(__file__).parent / "data" / "KPI_Suite_Data.csv"
    df = load_dataset(str(data_path))
    engine = KPIAnalyticsEngine(df)
    report = ReportGenerator(df, targets=engine.targets)

    st.title("📊 Enterprise KPI Reporting Suite")
    st.caption("Executive-style KPI reporting application demonstrating cross-functional performance monitoring.")
    st.caption(f"Author: {AUTHOR}")

    pages = [
        "Executive Overview", "Finance", "Sales", "People", "Operations",
        "Regional Performance", "Target & Variance Analysis", "Data Quality",
    ]
    page = st.sidebar.radio("View", pages)

    if page == "Executive Overview":
        summary = engine.generate_executive_summary()
        scope = summary["data_scope"]
        exceptions = engine.build_management_exceptions()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Revenue", f"$"+f"{scope['total_revenue']/1e6:.1f}M")
        c2.metric("Cost", f"$"+f"{scope['total_cost']/1e6:.1f}M")
        c3.metric("Margin", f"{scope['overall_margin_pct']:.1f}%")
        c4.metric("Management Exceptions", f"{len(exceptions):,}")

        st.subheader("Revenue and Cost Trend")
        st.plotly_chart(report.create_revenue_vs_cost_trend(), use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(report.create_quarterly_trend_chart(), use_container_width=True)
        with right:
            st.plotly_chart(report.create_status_breakdown_pie(), use_container_width=True)

        st.subheader("Current Management Exceptions")
        st.dataframe(exceptions.head(15), use_container_width=True, hide_index=True)

    elif page in {"Finance", "Sales", "People", "Operations"}:
        department = "HR" if page == "People" else page
        st.header(f"{department} Performance")
        show_target_table(engine, department)
        st.subheader("Department KPI Summary")
        dept = engine.calculate_department_kpis()
        if department in dept.index:
            st.dataframe(dept.loc[[department]], use_container_width=True)
        st.subheader("Exception Detail")
        exceptions = engine.build_management_exceptions()
        st.dataframe(exceptions[exceptions["Department"] == department],
                     use_container_width=True, hide_index=True)

    elif page == "Regional Performance":
        st.header("Regional Performance")
        st.plotly_chart(report.create_regional_performance_chart(), use_container_width=True)
        st.dataframe(engine.calculate_regional_performance(), use_container_width=True)
        st.subheader("Highest Below-Target Exposure")
        st.dataframe(pd.DataFrame(engine.identify_underperforming_areas(5)),
                     use_container_width=True, hide_index=True)

    elif page == "Target & Variance Analysis":
        st.header("Target & Variance Analysis")
        variance = engine.get_department_target_variance()
        st.plotly_chart(report.create_target_variance_heatmap(variance), use_container_width=True)
        st.subheader("Actual vs Target")
        show_target_table(engine)
        st.subheader("Illustrative Thresholds")
        st.caption("These thresholds are portfolio-analysis assumptions, not universal industry benchmarks.")
        for dept, metrics in engine.targets.items():
            with st.expander(dept):
                st.dataframe(pd.DataFrame(metrics).T.reset_index(names="KPI"),
                             use_container_width=True, hide_index=True)

    else:
        st.header("Data Quality & Privacy")
        loader = DataLoader(str(data_path))
        loader.load()
        quality = loader.check_data_quality()
        dims = quality["dimensions"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Overall", f"{quality['quality_score']:.1f}%")
        c2.metric("Completeness", f"{dims['completeness_pct']:.1f}%")
        c3.metric("Uniqueness", f"{dims['uniqueness_pct']:.1f}%")
        c4.metric("Validity", f"{dims['validity_pct']:.1f}%")
        c5.metric("Consistency", f"{dims['consistency_pct']:.1f}%")
        st.subheader("Quality Dimensions")
        st.json(quality)
        st.subheader("Employee ID Anonymization Demo")
        anon = DataAnonymizer(df.head(10).copy())
        original = df[["Employee_ID"]].head(10).copy()
        anon.hash_employee_id("Employee_ID")
        col1, col2 = st.columns(2)
        col1.dataframe(original, use_container_width=True, hide_index=True)
        col2.dataframe(anon.df[["Employee_ID"]], use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
