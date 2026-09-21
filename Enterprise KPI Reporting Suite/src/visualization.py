"""Visualization layer for the Enterprise KPI Reporting Suite.

Charts are based on actual metrics and centralized target definitions. No
cross-metric min-max normalization is used because relative magnitude is not
the same thing as target attainment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ReportGenerator:
    AUTHOR = "OGHENEOCHUKO EMMANUEL OGIDIAGBA"

    def __init__(self, df: pd.DataFrame, output_dir: str = "reports", targets=None):
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.targets = targets or {}
        self.figures: Dict[str, go.Figure] = {}

    def create_revenue_vs_cost_trend(self) -> go.Figure:
        data = self.df.copy()
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        trend = (
            data.dropna(subset=["Date"])
            .groupby("Date")[["Revenue", "Cost"]]
            .sum()
            .reset_index()
            .sort_values("Date")
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["Date"], y=trend["Revenue"],
                                 mode="lines", name="Revenue"))
        fig.add_trace(go.Scatter(x=trend["Date"], y=trend["Cost"],
                                 mode="lines", name="Cost"))
        fig.update_layout(
            title="Revenue vs Cost Over Time",
            xaxis_title="Date",
            yaxis_title="Amount ($)",
            hovermode="x unified",
            template="plotly_white",
            height=500,
        )
        self.figures["revenue_vs_cost_trend"] = fig
        return fig

    def create_target_variance_heatmap(self, variance_df: pd.DataFrame) -> go.Figure:
        """Heatmap of target-relative variance, where positive means better."""
        if variance_df.empty:
            fig = go.Figure()
        else:
            pivot = variance_df.pivot(index="KPI", columns="Department",
                                      values="relative_gap_pct")
            fig = px.imshow(
                pivot,
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                aspect="auto",
                labels={"color": "Gap vs Target (%)"},
                title="Target Variance by Department",
            )
            fig.update_traces(
                hovertemplate="<b>%{y}</b><br>%{x}<br>Gap vs target: %{z:.2f}%<extra></extra>"
            )
        fig.update_layout(template="plotly_white", height=500)
        self.figures["target_variance_heatmap"] = fig
        return fig

    # Backward-compatible name used by older callers.
    def create_department_heatmap(self) -> go.Figure:
        from src.kpi_engine import KPIAnalyticsEngine
        variance = KPIAnalyticsEngine(self.df).get_department_target_variance()
        return self.create_target_variance_heatmap(variance)

    def create_ltv_cac_distribution(self) -> go.Figure:
        fig = px.box(
            self.df,
            x="Department",
            y="LTV_CAC_Ratio",
            color="Department",
            title="LTV to CAC Ratio Distribution",
            labels={"LTV_CAC_Ratio": "LTV/CAC Ratio"},
        )
        sales_target = self.targets.get("Sales", {}).get("LTV_CAC_Ratio", {}).get("target")
        if sales_target is not None:
            fig.add_hline(
                y=sales_target,
                line_dash="dash",
                annotation_text=f"Illustrative target: {sales_target:g}",
            )
        fig.update_layout(height=500, template="plotly_white", showlegend=False)
        self.figures["ltv_cac_distribution"] = fig
        return fig

    def create_regional_performance_chart(self) -> go.Figure:
        regional = (
            self.df.groupby("Region")
            .agg(
                Revenue=("Revenue", "sum"),
                Cost=("Cost", "sum"),
                Gross_Margin=("Gross_Margin", "sum"),
                Record_Count=("Record_ID", "count"),
            )
            .reset_index()
        )
        regional["Margin_Pct"] = regional["Gross_Margin"] / regional["Revenue"] * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(x=regional["Region"], y=regional["Revenue"], name="Revenue"))
        fig.add_trace(go.Bar(x=regional["Region"], y=regional["Cost"], name="Cost"))
        fig.update_layout(
            title="Revenue and Cost by Region",
            xaxis_title="Region",
            yaxis_title="Amount ($)",
            barmode="group",
            template="plotly_white",
            height=450,
        )
        self.figures["regional_performance"] = fig
        return fig

    def create_quarterly_trend_chart(self) -> go.Figure:
        data = self.df.copy()
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        quarterly = (
            data.dropna(subset=["Date"])
            .assign(Period=lambda x: x["Date"].dt.to_period("Q").astype(str))
            .groupby("Period")[["Revenue", "Cost"]]
            .sum()
            .reset_index()
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=quarterly["Period"], y=quarterly["Revenue"],
                                 mode="lines+markers", name="Revenue"))
        fig.add_trace(go.Scatter(x=quarterly["Period"], y=quarterly["Cost"],
                                 mode="lines+markers", name="Cost"))
        fig.update_layout(
            title="Quarterly Revenue and Cost",
            xaxis_title="Quarter",
            yaxis_title="Amount ($)",
            hovermode="x unified",
            template="plotly_white",
            height=450,
        )
        self.figures["quarterly_trend"] = fig
        return fig

    def create_status_breakdown_pie(self) -> go.Figure:
        status = self.df["Status"].value_counts().rename_axis("Status").reset_index(name="Count")
        fig = px.pie(status, values="Count", names="Status",
                     title="Source Dataset Status Distribution")
        fig.update_layout(height=450, template="plotly_white")
        self.figures["status_breakdown"] = fig
        return fig

    def create_exception_priority_chart(self, exceptions: pd.DataFrame) -> go.Figure:
        if exceptions.empty:
            counts = pd.DataFrame({"Priority": ["No Exceptions"], "Count": [0]})
        else:
            counts = exceptions["Priority"].value_counts().reindex(
                ["High", "Medium", "Low"], fill_value=0
            ).rename_axis("Priority").reset_index(name="Count")
        fig = px.bar(
            counts, x="Priority", y="Count", title="Management Exceptions by Priority",
            text="Count"
        )
        fig.update_layout(template="plotly_white", height=350)
        self.figures["exception_priority"] = fig
        return fig

    def save_all_figures(self, prefix: str = "kpi_report") -> List[str]:
        saved_files = []
        for name, fig in self.figures.items():
            path = self.output_dir / f"{prefix}_{name}.html"
            fig.write_html(str(path))
            saved_files.append(str(path))
        return saved_files

    def generate_json_metrics(self, summary_data: Dict) -> str:
        payload = dict(summary_data)
        payload["author"] = self.AUTHOR
        payload["report_generated_at"] = pd.Timestamp.now().isoformat()
        path = self.output_dir / "executive_metrics.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return str(path)

    def create_executive_dashboard_data(self) -> Dict:
        return {
            "author": self.AUTHOR,
            "total_records": int(len(self.df)),
            "total_revenue": float(self.df["Revenue"].sum()),
            "total_cost": float(self.df["Cost"].sum()),
            "departments": sorted(self.df["Department"].dropna().unique().tolist()),
            "regions": sorted(self.df["Region"].dropna().unique().tolist()),
            "figures_available": list(self.figures.keys()),
        }
