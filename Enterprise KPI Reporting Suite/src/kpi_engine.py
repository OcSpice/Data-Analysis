"""KPI analytics engine for the Enterprise KPI Reporting Suite.

The module separates observed metrics, derived metrics, and illustrative
management thresholds. It intentionally avoids a composite "health score":
different KPIs have different units and business meanings, so performance is
reported as actual values, target gaps, and target status.
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KPIAnalyticsEngine:
    """Calculate enterprise KPIs, target variance, trends, and exceptions."""

    AUTHOR = "OGHENEOCHUKO EMMANUEL OGIDIAGBA"

    # These are portfolio/demo management thresholds, not universal benchmarks.
    DEPARTMENT_TARGETS = {
        "Finance": {
            "Margin_Pct": {
                "target": 40.0,
                "direction": "gte",
                "unit": "%",
                "description": "Gross margin percentage",
            },
            "Revenue_vs_Cost": {
                "target": 1.0,
                "direction": "gte",
                "unit": "ratio",
                "description": "Revenue-to-cost ratio",
            },
        },
        "Sales": {
            "LTV_CAC_Ratio": {
                "target": 3.0,
                "direction": "gte",
                "unit": "ratio",
                "description": "Lifetime value to customer acquisition cost",
            },
            "Conv_Rate_Pct": {
                "target": 15.0,
                "direction": "gte",
                "unit": "%",
                "description": "Lead-to-deal conversion rate",
            },
            "Deals_Closed": {
                "target": 5.0,
                "direction": "gte",
                "unit": "deals",
                "description": "Closed deals per reporting record",
            },
        },
        "HR": {
            "Attrition_Pct": {
                "target": 12.0,
                "direction": "lte",
                "unit": "%",
                "description": "Observed attrition percentage",
            },
            "Productivity_Pct": {
                "target": 70.0,
                "direction": "gte",
                "unit": "%",
                "description": "Productivity percentage",
            },
            "Training_Hrs": {
                "target": 20.0,
                "direction": "gte",
                "unit": "hours",
                "description": "Training hours per reporting record",
            },
        },
        "Operations": {
            "SLA_Met": {
                "target": 1.0,
                "direction": "gte",
                "unit": "rate",
                "description": "Share of records meeting SLA",
            },
            "Uptime_Pct": {
                "target": 99.5,
                "direction": "gte",
                "unit": "%",
                "description": "System/service uptime",
            },
            "Resolution_Hrs": {
                "target": 12.0,
                "direction": "lte",
                "unit": "hours",
                "description": "Average resolution time",
            },
        },
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.targets = self.DEPARTMENT_TARGETS
        self.exception_table: Optional[pd.DataFrame] = None

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if pd.isna(denominator) or denominator == 0:
            return np.nan
        return numerator / denominator

    @staticmethod
    def _weighted_margin(group: pd.DataFrame) -> float:
        revenue = group["Revenue"].sum()
        if revenue == 0:
            return np.nan
        return group["Gross_Margin"].sum() / revenue * 100

    def calculate_department_kpis(self) -> pd.DataFrame:
        """Aggregate department KPIs using appropriate sum/weighted/mean logic."""
        if self.df.empty:
            return pd.DataFrame()

        grouped = self.df.groupby("Department", dropna=False)
        rows = []

        for department, group in grouped:
            row = {
                "Revenue": group["Revenue"].sum(),
                "Cost": group["Cost"].sum(),
                "Gross_Margin": group["Gross_Margin"].sum(),
                "Margin_Pct": self._weighted_margin(group),
                "Deals_Closed": group["Deals_Closed"].mean(),
                "Leads_Generated": group["Leads_Generated"].sum(),
                "Conv_Rate_Pct": self._safe_ratio(
                    group["Deals_Closed"].sum(), group["Leads_Generated"].sum()
                ) * 100,
                "Customer_Sat": group["Customer_Sat"].mean(),
                "NPS": group["NPS"].mean(),
                "Headcount": group["Headcount"].mean(),
                "Attrition_Pct": group["Attrition_Pct"].mean(),
                "Training_Hrs": group["Training_Hrs"].mean(),
                "Productivity_Pct": group["Productivity_Pct"].mean(),
                "CAC": group["CAC"].mean(),
                "LTV": group["LTV"].mean(),
                "LTV_CAC_Ratio": self._safe_ratio(
                    group["LTV"].mean(), group["CAC"].mean()
                ),
                "MQL_Count": group["MQL_Count"].sum(),
                "SQL_Count": group["SQL_Count"].sum(),
                "CTR_Pct": group["CTR_Pct"].mean(),
                "SLA_Met": group["SLA_Met"].mean(),
                "Ticket_Volume": group["Ticket_Volume"].sum(),
                "Resolution_Hrs": group["Resolution_Hrs"].mean(),
                "Uptime_Pct": group["Uptime_Pct"].mean(),
                "Record_Count": len(group),
            }
            row["Revenue_vs_Cost"] = self._safe_ratio(row["Revenue"], row["Cost"])
            rows.append(pd.Series(row, name=department))

        return pd.DataFrame(rows).round(2)

    def calculate_regional_performance(self) -> pd.DataFrame:
        """Aggregate financial, service, and workforce metrics by region."""
        if self.df.empty:
            return pd.DataFrame()

        rows = []
        for region, group in self.df.groupby("Region", dropna=False):
            revenue = group["Revenue"].sum()
            cost = group["Cost"].sum()
            row = {
                "Revenue": revenue,
                "Cost": cost,
                "Gross_Margin": group["Gross_Margin"].sum(),
                "Margin_Pct": self._weighted_margin(group),
                "Deals_Closed": group["Deals_Closed"].sum(),
                "Attrition_Pct": group["Attrition_Pct"].mean(),
                "Productivity_Pct": group["Productivity_Pct"].mean(),
                "LTV_CAC_Ratio": self._safe_ratio(
                    group["LTV"].mean(), group["CAC"].mean()
                ),
                "SLA_Met": group["SLA_Met"].mean(),
                "Uptime_Pct": group["Uptime_Pct"].mean(),
                "Resolution_Hrs": group["Resolution_Hrs"].mean(),
                "Record_Count": len(group),
            }
            row["Revenue_vs_Cost"] = self._safe_ratio(revenue, cost)
            rows.append(pd.Series(row, name=region))

        return pd.DataFrame(rows).round(2)

    def calculate_quarterly_trends(self) -> pd.DataFrame:
        """Return quarterly totals plus period-over-period variance."""
        if self.df.empty:
            return pd.DataFrame()

        data = self.df.copy()
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data["Period"] = data["Date"].dt.to_period("Q")

        quarterly = (
            data.dropna(subset=["Period"])
            .groupby("Period")
            .agg(
                Revenue=("Revenue", "sum"),
                Cost=("Cost", "sum"),
                Gross_Margin=("Gross_Margin", "sum"),
                Deals_Closed=("Deals_Closed", "sum"),
                Attrition_Pct=("Attrition_Pct", "mean"),
                Productivity_Pct=("Productivity_Pct", "mean"),
                LTV_CAC_Ratio=("LTV_CAC_Ratio", "mean"),
                SLA_Met=("SLA_Met", "mean"),
                Uptime_Pct=("Uptime_Pct", "mean"),
                Resolution_Hrs=("Resolution_Hrs", "mean"),
                Record_Count=("Record_ID", "count"),
            )
            .reset_index()
        )
        quarterly["Margin_Pct"] = np.where(
            quarterly["Revenue"] != 0,
            quarterly["Gross_Margin"] / quarterly["Revenue"] * 100,
            np.nan,
        )
        quarterly["Revenue_vs_Cost"] = (
            quarterly["Revenue"] / quarterly["Cost"].replace(0, np.nan)
        )
        quarterly["Revenue_Change_Pct"] = quarterly["Revenue"].pct_change() * 100
        quarterly["Margin_Change_Pp"] = quarterly["Margin_Pct"].diff()
        quarterly["Period"] = quarterly["Period"].astype(str)
        quarterly["Year"] = quarterly["Period"].str[:4].astype(int)
        quarterly["Quarter"] = quarterly["Period"].str[-2:]

        return quarterly.round(2)

    @staticmethod
    def _evaluate(actual: float, target: float, direction: str) -> Dict[str, object]:
        """Evaluate an actual KPI against its target without cross-KPI scoring."""
        if pd.isna(actual) or pd.isna(target):
            return {
                "status": "No Data",
                "gap": np.nan,
                "attainment_pct": np.nan,
                "relative_gap_pct": np.nan,
            }

        gap = actual - target
        if direction == "gte":
            met = actual >= target
            attainment = (actual / target * 100) if target != 0 else np.nan
            relative_gap = (actual - target) / abs(target) * 100 if target else np.nan
        else:
            met = actual <= target
            attainment = (target / actual * 100) if actual != 0 else 100.0
            relative_gap = (target - actual) / abs(target) * 100 if target else np.nan

        return {
            "status": "Meets Target" if met else "Below Target",
            "gap": gap,
            "attainment_pct": attainment,
            "relative_gap_pct": relative_gap,
        }

    def get_department_target_variance(self, department: Optional[str] = None) -> pd.DataFrame:
        """Build a transparent actual-vs-target table for one or all departments."""
        dept_kpis = self.calculate_department_kpis()
        departments = [department] if department else list(dept_kpis.index)
        rows = []

        for dept in departments:
            if dept not in dept_kpis.index or dept not in self.targets:
                continue
            for metric, config in self.targets[dept].items():
                actual = (
                    dept_kpis.loc[dept, metric]
                    if metric in dept_kpis.columns
                    else np.nan
                )
                evaluation = self._evaluate(
                    actual, config["target"], config["direction"]
                )
                rows.append(
                    {
                        "Department": dept,
                        "KPI": metric,
                        "Actual": actual,
                        "Target": config["target"],
                        "Direction": config["direction"],
                        "Unit": config["unit"],
                        **evaluation,
                        "Description": config["description"],
                    }
                )

        return pd.DataFrame(rows).round(2)

    def build_management_exceptions(self) -> pd.DataFrame:
        """Return only target exceptions, with an explicit rule-based priority."""
        variance = self.get_department_target_variance()
        if variance.empty:
            self.exception_table = variance
            return variance

        exceptions = variance[variance["status"] == "Below Target"].copy()

        def priority(relative_gap: float) -> str:
            if pd.isna(relative_gap):
                return "Unclassified"
            severity = abs(relative_gap)
            if severity >= 10:
                return "High"
            if severity >= 5:
                return "Medium"
            return "Low"

        exceptions["Priority"] = exceptions["relative_gap_pct"].apply(priority)
        exceptions = exceptions.sort_values(
            ["Priority", "relative_gap_pct", "Department", "KPI"],
            ascending=[True, True, True, True],
        )
        self.exception_table = exceptions.reset_index(drop=True)
        return self.exception_table

    def identify_underperforming_areas(self, top_n: int = 3) -> List[Dict]:
        """Identify areas with the highest reported below-target exposure.

        This is exposure ranking, not a composite performance score.
        """
        if self.df.empty:
            return []

        area_frames = []
        for dimension in ["Cost_Center", "Region"]:
            grouped = self.df.groupby(dimension, dropna=False).agg(
                Revenue=("Revenue", "sum"),
                Cost=("Cost", "sum"),
                Margin_Pct=("Gross_Margin", lambda x: np.nan),
                Record_Count=("Record_ID", "count"),
                Below_Target_Count=("Status", lambda x: (x == "Below Target").sum()),
            ).reset_index()

            # Weighted margin is calculated separately because agg receives only margin values.
            margins = self.df.groupby(dimension).apply(self._weighted_margin)
            grouped["Margin_Pct"] = grouped[dimension].map(margins)
            grouped["Below_Target_Rate"] = (
                grouped["Below_Target_Count"] / grouped["Record_Count"] * 100
            )
            grouped["Revenue_vs_Cost"] = (
                grouped["Revenue"] / grouped["Cost"].replace(0, np.nan)
            )
            grouped["Type"] = dimension
            grouped["Identifier"] = grouped[dimension]
            area_frames.append(grouped)

        combined = pd.concat(area_frames, ignore_index=True)
        combined = combined.sort_values(
            ["Below_Target_Rate", "Below_Target_Count"],
            ascending=[False, False],
        ).head(top_n)

        return combined[
            [
                "Type",
                "Identifier",
                "Record_Count",
                "Below_Target_Count",
                "Below_Target_Rate",
                "Revenue",
                "Cost",
                "Revenue_vs_Cost",
                "Margin_Pct",
            ]
        ].round(2).to_dict("records")

    def get_below_target_records(self) -> pd.DataFrame:
        """Return records carrying the source dataset's Below Target status."""
        return self.df[self.df["Status"] == "Below Target"].copy()

    def generate_executive_summary(self) -> Dict:
        """Generate an auditable summary without a composite health score."""
        total_revenue = self.df["Revenue"].sum()
        total_cost = self.df["Cost"].sum()
        total_margin = self.df["Gross_Margin"].sum()
        overall_margin_pct = (
            total_margin / total_revenue * 100 if total_revenue else np.nan
        )
        below_target_count = int((self.df["Status"] == "Below Target").sum())
        exceptions = self.build_management_exceptions()

        return {
            "author": self.AUTHOR,
            "generated_at": datetime.now().isoformat(),
            "data_scope": {
                "total_records": int(len(self.df)),
                "total_revenue": round(total_revenue, 2),
                "total_cost": round(total_cost, 2),
                "total_margin": round(total_margin, 2),
                "overall_margin_pct": round(overall_margin_pct, 2),
            },
            "performance": {
                "reported_below_target_records": below_target_count,
                "reported_below_target_percentage": round(
                    below_target_count / len(self.df) * 100, 2
                ) if len(self.df) else 0.0,
                "management_exception_count": int(len(exceptions)),
            },
            "departments_analyzed": sorted(self.df["Department"].dropna().unique().tolist()),
            "regions_analyzed": sorted(self.df["Region"].dropna().unique().tolist()),
        }

    def get_kpi_targets_reference(self) -> Dict:
        """Return documented illustrative thresholds used by the analysis."""
        return {
            "author": self.AUTHOR,
            "target_type": "Illustrative management thresholds for portfolio analysis",
            "targets": self.targets,
        }
