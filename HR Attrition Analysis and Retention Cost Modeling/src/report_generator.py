"""
Dynamic reporting for the HR attrition analysis.

All reported metrics are generated from the current pipeline results. No
business metrics or savings figures are hard-coded in this module.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class ReportGenerator:
    """Generate reproducible text, CSV, JSON and visualization outputs."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_reports(
        self,
        df_enriched: pd.DataFrame,
        insights: Dict[str, Any],
        model_results: Dict[str, Any],
        impact: Dict[str, Any],
        validation_report: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._generate_executive_summary(
            df_enriched, insights, model_results, impact, validation_report
        )
        self._generate_retention_plan_report(impact)
        self._generate_data_dictionary(df_enriched)
        self._generate_key_metrics_json(
            df_enriched, insights, model_results, impact, validation_report
        )
        self._generate_segment_csvs(df_enriched)
        self._generate_model_comparison_csv(model_results)

        try:
            self._generate_visualizations(df_enriched, insights, model_results, impact)
        except ImportError:
            print("Matplotlib is not installed; visualization generation skipped.")

    def _generate_executive_summary(
        self,
        df: pd.DataFrame,
        insights: Dict[str, Any],
        model_results: Dict[str, Any],
        impact: Dict[str, Any],
        validation_report: Optional[Dict[str, Any]],
    ) -> None:
        overtime = insights.get("attrition_by_overtime", {})
        rf = model_results.get("models", {}).get("random_forest", {})
        lr = model_results.get("models", {}).get("logistic_regression", {})
        top_features = model_results.get("top_features", {})

        lines = [
            "HR ATTRITION ANALYSIS AND RETENTION COST MODELING",
            "Executive Summary",
            "=" * 72,
            "",
            "DATASET",
            f"- Employee records: {len(df):,}",
            "",
            "WORKFORCE FINDINGS",
            f"- Overall observed attrition rate: {insights.get('overall_attrition_rate', 0):.1%}",
            f"- Overtime attrition rate: {overtime.get('Yes', 0):.1%}",
            f"- Non-overtime attrition rate: {overtime.get('No', 0):.1%}",
            f"- Overtime/non-overtime observed attrition-rate ratio: {insights.get('overtime_risk_ratio', 1):.2f}x",
            "",
            "PREDICTIVE MODELING",
            "Metrics below are evaluated on a held-out test set.",
            f"- Logistic Regression — ROC-AUC: {lr.get('roc_auc', 0):.3f}; PR-AUC: {lr.get('pr_auc', 0):.3f}; "
            f"Recall: {lr.get('recall', 0):.3f}; F1: {lr.get('f1', 0):.3f}",
            f"- Random Forest — ROC-AUC: {rf.get('roc_auc', 0):.3f}; PR-AUC: {rf.get('pr_auc', 0):.3f}; "
            f"Recall: {rf.get('recall', 0):.3f}; F1: {rf.get('f1', 0):.3f}",
            "",
            "MODEL EXPLAINABILITY",
        ]

        for idx, (feature, value) in enumerate(list(top_features.items())[:5], 1):
            lines.append(f"- {idx}. {feature}: mean absolute SHAP value = {value:.5f}")

        lines.extend([
            "",
            "FINANCIAL IMPACT",
            f"- Observed replacement-cost exposure: {impact.get('total_replacement_cost', 0):,.0f} USD",
            f"- Replacement-cost multiplier assumption: {impact.get('replacement_cost_multiplier', 1.5):.2f}x annual salary",
            "",
            "ILLUSTRATIVE RETENTION SCENARIOS",
        ])

        for scenario in impact.get("retention_scenarios", []):
            lines.append(
                f"- {scenario['retention_effectiveness']:.0%} effectiveness: "
                f"{scenario['avoided_cost_estimate']:,.0f} USD avoided-cost estimate"
            )

        lines.extend([
            "",
            "LIMITATIONS",
            "- The dataset is a benchmark HR attrition dataset and should not be treated as representative of every workforce.",
            "- Observational associations do not establish causation.",
            "- Replacement-cost values depend on the stated salary multiplier assumption.",
            "- Retention scenario values are hypothetical estimates, not observed or guaranteed savings.",
            "- Model predictions are probabilistic and should not be used as automatic employment decisions.",
            "",
            "DATA PRIVACY",
            "- EmployeeNumber is masked before analysis outputs are generated.",
        ])

        path = self.output_dir / "executive_summary.txt"
        path.write_text("\\n".join(lines), encoding="utf-8")
        print(f"Generated: {path}")

    def _generate_retention_plan_report(self, impact: Dict[str, Any]) -> None:
        lines = [
            "HR ATTRITION — RETENTION ACTION PLAN",
            "=" * 72,
            "",
            "The actions below are evidence-linked investigation and retention ideas.",
            "They are not presented as causal prescriptions or guaranteed savings.",
            "",
        ]

        for item in impact.get("retention_plan", []):
            lines.extend([
                f"{item['priority']}. {item['initiative']}",
                "-" * 72,
                f"Rationale: {item['rationale']}",
                f"Action: {item['action']}",
                "Metrics:",
            ])
            lines.extend([f"  - {metric}" for metric in item.get("metrics", [])])
            lines.append(f"Evidence note: {item['evidence_note']}")
            lines.append("")

        path = self.output_dir / "retention_plan.txt"
        path.write_text("\\n".join(lines), encoding="utf-8")
        print(f"Generated: {path}")

    def _generate_data_dictionary(self, df: pd.DataFrame) -> None:
        lines = [
            "HR ATTRITION DATA DICTIONARY",
            "=" * 72,
            "",
            "Column | Data type | Unique values | Missing values",
            "--- | --- | ---: | ---:",
        ]

        for col in df.columns:
            lines.append(
                f"{col} | {df[col].dtype} | {df[col].nunique(dropna=True)} | {df[col].isna().sum()}"
            )

        lines.extend([
            "",
            "ENGINEERED FEATURES",
            "- AnnualIncome = MonthlyIncome × 12",
            "- ReplacementCost = AnnualIncome × configured replacement-cost multiplier",
            "- TenureBucket = YearsAtCompany grouped into tenure bands",
            "- IsOverTime = binary indicator derived from OverTime",
            "- PromotionStagnation = YearsSinceLastPromotion >= 3",
            "- LowSatisfactionCount = count of low satisfaction indicators",
            "- PoorWorkLifeBalance = WorkLifeBalance <= 2",
        ])

        path = self.output_dir / "data_dictionary.txt"
        path.write_text("\\n".join(lines), encoding="utf-8")
        print(f"Generated: {path}")

    def _generate_key_metrics_json(
        self,
        df: pd.DataFrame,
        insights: Dict[str, Any],
        model_results: Dict[str, Any],
        impact: Dict[str, Any],
        validation_report: Optional[Dict[str, Any]],
    ) -> None:
        payload = {
            "portfolio_category": "Data Analysis",
            "project_name": "HR Attrition Analysis and Retention Cost Modeling",
            "dataset": {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
            },
            "data_quality": validation_report or {},
            "workforce_analysis": {
                "overall_attrition_rate": insights.get("overall_attrition_rate"),
                "attrition_by_overtime": insights.get("attrition_by_overtime"),
                "overtime_rate_ratio": insights.get("overtime_risk_ratio"),
                "attrition_by_department": insights.get("attrition_by_department"),
                "attrition_by_job_role": insights.get("attrition_by_job_role"),
                "attrition_by_tenure": insights.get("attrition_by_tenure"),
            },
            "model_performance": model_results.get("models", {}),
            "random_forest_top_features_shap": model_results.get("top_features", {}),
            "business_impact": impact,
        }

        path = self.output_dir / "key_metrics.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Generated: {path}")

    def _generate_segment_csvs(self, df: pd.DataFrame) -> None:
        segments = {
            "attrition_by_department.csv": "Department",
            "attrition_by_job_role.csv": "JobRole",
            "attrition_by_overtime.csv": "OverTime",
            "attrition_by_tenure.csv": "TenureBucket",
        }

        for filename, column in segments.items():
            if column not in df.columns:
                continue
            grouped = (
                df.groupby(column, dropna=False)["Attrition"]
                .agg(
                    employee_count="size",
                    attrition_count=lambda s: (s == "Yes").sum(),
                )
                .reset_index()
            )
            grouped["attrition_rate"] = (
                grouped["attrition_count"] / grouped["employee_count"]
            )
            grouped.to_csv(self.output_dir / filename, index=False)

        print(f"Generated: segment CSVs in {self.output_dir}")

    def _generate_model_comparison_csv(self, model_results: Dict[str, Any]) -> None:
        rows = []
        for model_name, metrics in model_results.get("models", {}).items():
            rows.append({
                "model": model_name,
                "accuracy": metrics.get("accuracy"),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
            })

        pd.DataFrame(rows).to_csv(
            self.output_dir / "model_comparison.csv", index=False
        )

    def _generate_visualizations(
        self,
        df: pd.DataFrame,
        insights: Dict[str, Any],
        model_results: Dict[str, Any],
        impact: Dict[str, Any],
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        overtime = insights.get("attrition_by_overtime", {})
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            ["No Overtime", "Overtime"],
            [overtime.get("No", 0) * 100, overtime.get("Yes", 0) * 100],
        )
        ax.set_ylabel("Observed attrition rate (%)")
        ax.set_title("Observed Attrition Rate by Overtime Status")
        fig.tight_layout()
        fig.savefig(self.output_dir / "fig1_attrition_by_overtime.png", dpi=150)
        plt.close(fig)

        if "Department" in df.columns:
            dept = (
                df.groupby("Department")["Attrition"]
                .apply(lambda s: (s == "Yes").mean())
                .sort_values(ascending=False)
            )
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(dept.index.astype(str), dept.values * 100)
            ax.set_ylabel("Observed attrition rate (%)")
            ax.set_title("Observed Attrition Rate by Department")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(self.output_dir / "fig2_attrition_by_department.png", dpi=150)
            plt.close(fig)

        top = model_results.get("top_features", {})
        if top:
            features = list(top.keys())[:10][::-1]
            values = [top[x] for x in features]
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(features, values)
            ax.set_xlabel("Mean absolute SHAP value")
            ax.set_title("Top Random Forest Features by SHAP Importance")
            fig.tight_layout()
            fig.savefig(self.output_dir / "fig3_shap_features.png", dpi=150)
            plt.close(fig)

        scenarios = impact.get("retention_scenarios", [])
        if scenarios:
            labels = [f"{s['retention_effectiveness']:.0%}" for s in scenarios]
            values = [s["avoided_cost_estimate"] / 1e6 for s in scenarios]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(labels, values)
            ax.set_xlabel("Illustrative retention effectiveness")
            ax.set_ylabel("Avoided-cost estimate (USD millions)")
            ax.set_title("Retention Scenario Sensitivity")
            fig.tight_layout()
            fig.savefig(self.output_dir / "fig4_retention_scenarios.png", dpi=150)
            plt.close(fig)

        print(f"Generated: visualization files in {self.output_dir}")
