"""Data loading and data-quality checks for the KPI reporting suite."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load the source CSV and produce transparent quality dimensions."""

    EXPECTED_COLUMNS = [
        "Record_ID", "Date", "Year", "Month", "Quarter", "Department",
        "Region", "Product", "Channel", "Employee_ID", "Cost_Center",
        "Revenue", "Cost", "Gross_Margin", "Margin_Pct", "Deals_Closed",
        "Leads_Generated", "Conv_Rate_Pct", "Customer_Sat", "NPS",
        "Headcount", "Attrition_Pct", "Training_Hrs", "Productivity_Pct",
        "CAC", "LTV", "LTV_CAC_Ratio", "MQL_Count", "SQL_Count", "CTR_Pct",
        "SLA_Met", "Ticket_Volume", "Resolution_Hrs", "Uptime_Pct", "Status",
    ]

    NON_NEGATIVE_COLUMNS = [
        "Revenue", "Cost", "Gross_Margin", "Deals_Closed", "Leads_Generated",
        "Headcount", "Training_Hrs", "CAC", "LTV", "MQL_Count", "SQL_Count",
        "Ticket_Volume", "Resolution_Hrs",
    ]

    PERCENTAGE_COLUMNS = [
        "Margin_Pct", "Conv_Rate_Pct", "Customer_Sat", "Attrition_Pct",
        "Productivity_Pct", "CTR_Pct", "Uptime_Pct",
    ]

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.validation_errors: List[str] = []

    def load(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        return self.df

    def validate_schema(self) -> Tuple[bool, List[str]]:
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")
        missing = sorted(set(self.EXPECTED_COLUMNS) - set(self.df.columns))
        if missing:
            self.validation_errors.append(f"Missing columns: {missing}")
        return not missing, missing

    def check_data_quality(self) -> dict:
        """Return completeness, validity, uniqueness and consistency separately."""
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")

        df = self.df
        total_cells = max(df.shape[0] * df.shape[1], 1)

        missing_cells = int(df.isna().sum().sum())
        completeness = max(0.0, 100 * (1 - missing_cells / total_cells))

        duplicate_records = int(df.duplicated().sum())
        unique_record_ids = (
            int(df["Record_ID"].nunique()) if "Record_ID" in df.columns else 0
        )
        uniqueness_denominator = max(len(df), 1)
        uniqueness = 100 * unique_record_ids / uniqueness_denominator

        validity_issues = {}
        for col in self.NON_NEGATIVE_COLUMNS:
            if col in df.columns:
                count = int((pd.to_numeric(df[col], errors="coerce") < 0).sum())
                if count:
                    validity_issues[f"negative_{col}"] = count

        if {"Revenue", "Cost"}.issubset(df.columns):
            validity_issues["revenue_below_cost"] = int(
                (df["Revenue"] < df["Cost"]).sum()
            )

        for col in self.PERCENTAGE_COLUMNS:
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                count = int(((numeric < 0) | (numeric > 100)).sum())
                if count:
                    validity_issues[f"out_of_range_{col}"] = count

        if "SLA_Met" in df.columns:
            validity_issues["invalid_sla_flag"] = int(
                (~df["SLA_Met"].isin([0, 1])).sum()
            )

        validity_issue_count = sum(validity_issues.values())
        validity = max(
            0.0,
            100 * (1 - validity_issue_count / max(len(df), 1)),
        )

        consistency_issues = {}
        if {"Revenue", "Cost", "Gross_Margin"}.issubset(df.columns):
            expected_margin = df["Revenue"] - df["Cost"]
            consistency_issues["margin_mismatch"] = int(
                (~(abs(df["Gross_Margin"] - expected_margin) <= 0.02)).sum()
            )

        if {"Gross_Margin", "Revenue", "Margin_Pct"}.issubset(df.columns):
            expected_pct = (
                df["Gross_Margin"] / df["Revenue"].replace(0, pd.NA) * 100
            )
            consistency_issues["margin_pct_mismatch"] = int(
                (~(abs(df["Margin_Pct"] - expected_pct) <= 0.2)).fillna(False).sum()
            )

        if {"LTV", "CAC", "LTV_CAC_Ratio"}.issubset(df.columns):
            expected_ratio = df["LTV"] / df["CAC"].replace(0, pd.NA)
            consistency_issues["ltv_cac_mismatch"] = int(
                (~(abs(df["LTV_CAC_Ratio"] - expected_ratio) <= 0.05)).fillna(False).sum()
            )

        consistency_issue_count = sum(consistency_issues.values())
        consistency = max(
            0.0,
            100 * (1 - consistency_issue_count / max(len(df), 1)),
        )

        # Composite is an equal-weight summary of four explicitly named dimensions.
        overall = (completeness + uniqueness + validity + consistency) / 4

        report = {
            "total_records": int(len(df)),
            "total_columns": int(len(df.columns)),
            "missing_values": {
                col: int(count) for col, count in df.isna().sum().items() if count
            },
            "duplicate_records": duplicate_records,
            "invalid_revenue_cost": int(
                validity_issues.get("revenue_below_cost", 0)
            ),
            "negative_values": {
                key.replace("negative_", ""): value
                for key, value in validity_issues.items()
                if key.startswith("negative_")
            },
            "validity_issues": validity_issues,
            "consistency_issues": consistency_issues,
            "dimensions": {
                "completeness_pct": round(completeness, 2),
                "uniqueness_pct": round(uniqueness, 2),
                "validity_pct": round(validity, 2),
                "consistency_pct": round(consistency, 2),
            },
            "quality_score": round(overall, 2),
        }
        return report

    def handle_missing_values(self, strategy: str = "drop") -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")

        if strategy == "drop":
            self.df = self.df.dropna()
        elif strategy in {"fill_mean", "fill_median"}:
            numeric_cols = self.df.select_dtypes(include="number").columns
            fill_values = (
                self.df[numeric_cols].mean()
                if strategy == "fill_mean"
                else self.df[numeric_cols].median()
            )
            self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_values)
        elif strategy == "fill_zero":
            self.df = self.df.fillna(0)
        else:
            raise ValueError(
                "strategy must be one of: drop, fill_mean, fill_median, fill_zero"
            )
        return self.df
