"""
Streamlit Dashboard for Enterprise KPI Reporting Suite
Interactive CFO-level dashboard for executive performance tracking.

Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import DataLoader
from src.kpi_engine import KPIAnalyticsEngine
from src.visualization import ReportGenerator
from src.anonymizer import DataAnonymizer

# Page configuration
st.set_page_config(
    page_title="Enterprise KPI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Author metadata constant
AUTHOR = "OGHENEOCHUKO EMMANUEL OGIDIAGBA"


def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare data using DataLoader."""
    loader = DataLoader(data_path)
    df = loader.load()
    loader.validate_schema()
    return df


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("📊 Enterprise KPI Reporting Suite")
    st.markdown(f"**Author:** {AUTHOR}")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View:",
        ["Executive Overview", "Department Analysis", "Regional Performance", 
         "Target Achievement", "Data Quality"]
    )
    
    # Load data
    data_path = Path(__file__).parent / 'data' / 'KPI_Suite_Data.csv'
    
    try:
        df = load_data(str(data_path))
    except FileNotFoundError:
        st.error("Data file not found. Please ensure KPI_Suite_Data.csv exists in the data folder.")
        return
    
    # Initialize engines
    kpi_engine = KPIAnalyticsEngine(df)
    report_gen = ReportGenerator(df)
    
    # Calculate key metrics
    total_revenue = df['Revenue'].sum()
    total_cost = df['Cost'].sum()
    total_margin = df['Gross_Margin'].sum()
    margin_pct = (total_margin / total_revenue * 100) if total_revenue > 0 else 0
    
    below_target = (df['Status'] == 'Below Target').sum()
    above_target = len(df) - below_target
    
    # Executive Overview Page
    if page == "Executive Overview":
        st.header("Executive Overview")
        st.markdown(f"*Tracking enterprise performance across ${total_revenue/1e9:.1f} billion revenue scope*")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Revenue",
                value=f"${total_revenue/1e6:.1f}M",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Total Cost",
                value=f"${total_cost/1e6:.1f}M",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Gross Margin",
                value=f"{margin_pct:.1f}%",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Records Analyzed",
                value=f"{len(df):,}",
                delta=None
            )
        
        st.markdown("---")
        
        # Two column layout for charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Revenue vs Cost trend
            fig_trend = report_gen.create_revenue_vs_cost_trend()
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col_right:
            # Status breakdown
            fig_pie = report_gen.create_status_breakdown_pie()
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Quarterly trends
        st.subheader("Quarterly Performance Trends")
        fig_quarterly = report_gen.create_quarterly_trend_chart()
        st.plotly_chart(fig_quarterly, use_container_width=True)
        
        # Department health scores
        st.subheader("Department Health Scores")
        health_scores = kpi_engine.calculate_all_health_scores()
        
        health_cols = st.columns(len(health_scores))
        for idx, (dept, score) in enumerate(health_scores.items()):
            with health_cols[idx]:
                st.metric(
                    label=dept,
                    value=f"{score:.0f}/100",
                    delta=None
                )
    
    # Department Analysis Page
    elif page == "Department Analysis":
        st.header("Department Analysis")
        
        # Department selector
        departments = sorted(df['Department'].unique())
        selected_dept = st.selectbox("Select Department:", departments)
        
        dept_data = df[df['Department'] == selected_dept]
        
        # Department metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Department Revenue",
                f"${dept_data['Revenue'].sum()/1e6:.1f}M"
            )
        
        with col2:
            st.metric(
                "Avg Margin %",
                f"{dept_data['Margin_Pct'].mean():.1f}%"
            )
        
        with col3:
            below_count = (dept_data['Status'] == 'Below Target').sum()
            st.metric(
                "Below Target Records",
                f"{below_count:,}"
            )
        
        st.markdown("---")
        
        # LTV/CAC distribution for selected department
        st.subheader(f"LTV to CAC Ratio - {selected_dept}")
        fig_ltv = report_gen.create_ltv_cac_distribution()
        st.plotly_chart(fig_ltv, use_container_width=True)
        
        # Department KPI table
        st.subheader("Department KPI Summary")
        dept_kpis = kpi_engine.calculate_department_kpis()
        st.dataframe(dept_kpis.loc[[selected_dept]], use_container_width=True)
    
    # Regional Performance Page
    elif page == "Regional Performance":
        st.header("Regional Performance Analysis")
        
        # Regional performance chart
        fig_regional = report_gen.create_regional_performance_chart()
        st.plotly_chart(fig_regional, use_container_width=True)
        
        # Regional data table
        st.subheader("Regional Metrics")
        regional_data = kpi_engine.calculate_regional_performance()
        st.dataframe(regional_data, use_container_width=True)
        
        # Underperforming areas
        st.subheader("Underperforming Areas")
        underperforming = kpi_engine.identify_underperforming_areas(top_n=5)
        
        if underperforming:
            up_df = pd.DataFrame(underperforming)
            st.dataframe(up_df, use_container_width=True)
    
    # Target Achievement Page
    elif page == "Target Achievement":
        st.header("Target Achievement Heatmap")
        
        # Heatmap
        fig_heatmap = report_gen.create_department_heatmap()
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("---")
        
        # KPI Targets Reference
        st.subheader("Business Targets Reference")
        targets_ref = kpi_engine.get_kpi_targets_reference()
        
        for dept, metrics in targets_ref['targets'].items():
            with st.expander(f"{dept} Targets"):
                for metric, info in metrics.items():
                    direction_symbol = "≥" if info['direction'] == 'gte' else "≤"
                    st.write(f"- **{metric}**: {direction_symbol} {info['target']}")
        
        # Below target records detail
        st.subheader("Below Target Records Detail")
        below_df = kpi_engine.get_below_target_records()
        
        if st.button("Show Below Target Records"):
            st.dataframe(below_df.head(100), use_container_width=True)
            st.write(f"*Showing first 100 of {len(below_df)} below-target records*")
    
    # Data Quality Page
    elif page == "Data Quality":
        st.header("Data Quality & Privacy")
        
        # Run quality checks
        loader = DataLoader(str(data_path))
        loader.load()
        quality_report = loader.check_data_quality()
        
        # Quality score
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Quality Score", f"{quality_report['quality_score']}%")
        
        with col2:
            st.metric("Total Records", f"{quality_report['total_records']:,}")
        
        st.markdown("---")
        
        # Missing values
        st.subheader("Missing Values by Column")
        if quality_report['missing_values']:
            missing_df = pd.DataFrame(list(quality_report['missing_values'].items()),
                                     columns=['Column', 'Missing Count'])
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values detected!")
        
        # Data validation issues
        st.subheader("Validation Issues")
        if quality_report['invalid_revenue_cost'] > 0:
            st.warning(f"{quality_report['invalid_revenue_cost']} records have Revenue < Cost")
        else:
            st.success("All records pass Revenue >= Cost validation")
        
        if quality_report['duplicate_records'] > 0:
            st.warning(f"{quality_report['duplicate_records']} duplicate records found")
        else:
            st.success("No duplicate records found")
        
        # Anonymization demo
        st.markdown("---")
        st.subheader("Data Anonymization Demo")
        
        anon = DataAnonymizer(df.head(10).copy())
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write("**Original Employee IDs:**")
            st.dataframe(df[['Employee_ID']].head(10), use_container_width=True)
        
        with col_b:
            anon.hash_employee_id()
            st.write("**Anonymized Employee IDs:**")
            st.dataframe(anon.df[['Employee_ID']].head(10), use_container_width=True)
        
        anon_report = anon.get_anonymization_report()
        st.info(f"Anonymization method: {anon_report['method_used']}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: gray;'>
            <p><strong>Enterprise KPI Reporting Suite</strong></p>
            <p>Portfolio Project - Data Analysis Track</p>
            <p>Author: {AUTHOR}</p>
            <p>Data Scope: ~38,000 records | $1.4 billion revenue scope</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
