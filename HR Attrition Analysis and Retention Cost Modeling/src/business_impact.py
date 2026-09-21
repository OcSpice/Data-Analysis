"""
Business impact analysis for HR attrition.

The module separates observed replacement-cost exposure from hypothetical
retention scenarios. It does not claim that a retention intervention caused
or will guarantee a specific amount of savings.
"""

from typing import Any, Dict, List

import pandas as pd


class BusinessImpactAnalyzer:
    """Calculate replacement-cost exposure and retention scenarios."""

    def __init__(
        self,
        replacement_cost_multiplier: float = 1.5,
        scenario_effectiveness: List[float] | None = None,
        high_risk_threshold: float = 0.50,
    ):
        if replacement_cost_multiplier <= 0:
            raise ValueError("replacement_cost_multiplier must be positive")

        self.replacement_cost_multiplier = replacement_cost_multiplier
        self.scenario_effectiveness = scenario_effectiveness or [0.10, 0.20, 0.30, 0.40, 0.50]
        if any(rate < 0 or rate > 1 for rate in self.scenario_effectiveness):
            raise ValueError("Scenario effectiveness values must be between 0 and 1")

        if high_risk_threshold < 0 or high_risk_threshold > 1:
            raise ValueError("high_risk_threshold must be between 0 and 1")
        self.high_risk_threshold = high_risk_threshold

    def calculate_impact(
        self,
        df: pd.DataFrame,
        insights: Dict[str, Any],
        model_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate observed exposure, scenarios and model-risk diagnostics."""
        total_exposure = self._calculate_total_replacement_cost(df)
        scenarios = self._calculate_retention_scenarios(total_exposure)

        return {
            "total_replacement_cost": total_exposure,
            "annual_exposure": self._calculate_annual_exposure(df),
            "replacement_cost_multiplier": self.replacement_cost_multiplier,
            "retention_scenarios": scenarios,
            "high_risk_employees": self._identify_high_risk_employees(df, model_results),
            "retention_plan": self._generate_retention_plan(insights, model_results),
            "financial_methodology": {
                "exposure_definition": (
                    "Observed replacement-cost estimate for employees recorded "
                    "as Attrition=Yes."
                ),
                "scenario_definition": (
                    "Illustrative avoided-cost scenarios assuming the stated "
                    "share of observed replacement-cost exposure could be avoided."
                ),
                "causality": "Scenarios are not observed savings or causal estimates.",
            },
        }

    def _calculate_total_replacement_cost(self, df: pd.DataFrame) -> float:
        """Sum replacement-cost estimates for observed departures."""
        if "Attrition" not in df.columns:
            raise ValueError("DataFrame must contain 'Attrition'")
        if "AnnualIncome" not in df.columns:
            raise ValueError("DataFrame must contain 'AnnualIncome'")

        departed = df[df["Attrition"].astype(str).str.lower() == "yes"]
        return float(
            (departed["AnnualIncome"] * self.replacement_cost_multiplier).sum()
        )

    def _calculate_annual_exposure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate descriptive replacement-cost exposure by workforce segment."""
        if "AnnualIncome" not in df.columns:
            return {}

        working = df.copy()
        working["ReplacementCostModel"] = (
            working["AnnualIncome"] * self.replacement_cost_multiplier
        )

        exposure: Dict[str, Any] = {
            "total_workforce_replacement_value": float(
                working["ReplacementCostModel"].sum()
            )
        }

        if "OverTime" in working.columns:
            exposure["by_overtime"] = {
                str(k): float(v)
                for k, v in working.groupby("OverTime")["ReplacementCostModel"].sum().items()
            }

        if "Department" in working.columns:
            exposure["by_department"] = {
                str(k): float(v)
                for k, v in working.groupby("Department")["ReplacementCostModel"].sum().items()
            }

        return exposure

    def _calculate_retention_scenarios(self, total_exposure: float) -> List[Dict[str, float]]:
        """Calculate illustrative avoided-cost scenarios from observed exposure."""
        scenarios = []
        for effectiveness in self.scenario_effectiveness:
            avoided = total_exposure * effectiveness
            scenarios.append({
                "retention_effectiveness": effectiveness,
                "avoided_cost_estimate": float(avoided),
                "remaining_exposure_estimate": float(total_exposure - avoided),
            })
        return scenarios

    def _identify_high_risk_employees(
        self,
        df: pd.DataFrame,
        model_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize high predicted risk on held-out test observations."""
        probabilities = model_results.get("test_predicted_probabilities")
        if probabilities is None:
            return {
                "definition": "Unavailable",
                "count": 0,
                "percentage": 0.0,
                "note": "No held-out prediction probabilities were supplied.",
            }

        probabilities = pd.Series(probabilities)
        high_risk = probabilities >= self.high_risk_threshold

        return {
            "definition": (
                f"Held-out test observations with predicted attrition probability "
                f">= {self.high_risk_threshold:.0%}"
            ),
            "count": int(high_risk.sum()),
            "percentage": float(high_risk.mean() * 100),
            "threshold": self.high_risk_threshold,
            "note": (
                "This is a model-predicted segment, not an observed employee "
                "outcome and not a causal risk classification."
            ),
        }

    def _generate_retention_plan(
        self,
        insights: Dict[str, Any],
        model_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate cautious, evidence-linked retention actions."""
        overtime_ratio = insights.get("overtime_risk_ratio", 1.0)
        top_features = list(model_results.get("top_features", {}).keys())
        feature_text = ", ".join(top_features[:4]) if top_features else "model features"

        return [
            {
                "priority": 1,
                "initiative": "Review overtime workload and scheduling",
                "rationale": (
                    f"The observed attrition rate among overtime employees was "
                    f"{overtime_ratio:.2f} times the rate among non-overtime employees."
                ),
                "action": (
                    "Review overtime frequency, staffing coverage, workload allocation "
                    "and voluntary versus mandatory overtime."
                ),
                "metrics": [
                    "Overtime participation rate",
                    "Average overtime hours",
                    "Attrition rate by overtime status",
                ],
                "evidence_note": "Association in the observed dataset; not proof of causation.",
            },
            {
                "priority": 2,
                "initiative": "Strengthen career progression monitoring",
                "rationale": (
                    "Use tenure and promotion-related patterns identified in the "
                    "descriptive analysis and model explanations."
                ),
                "action": (
                    "Review employees approaching long promotion intervals and "
                    "document clearer progression criteria."
                ),
                "metrics": [
                    "Median years since last promotion",
                    "Internal promotion rate",
                    "Attrition rate by promotion-stagnation group",
                ],
                "evidence_note": f"Model explanation features include: {feature_text}.",
            },
            {
                "priority": 3,
                "initiative": "Target early-tenure retention interventions",
                "rationale": (
                    "Compare attrition rates across tenure bands and focus attention "
                    "where observed rates are materially higher."
                ),
                "action": (
                    "Use structured onboarding, manager check-ins and development "
                    "plans at higher-risk tenure milestones."
                ),
                "metrics": [
                    "Attrition rate by tenure band",
                    "90-day and 12-month retention",
                    "New-hire satisfaction",
                ],
                "evidence_note": "Use observed segment differences to prioritize investigation.",
            },
            {
                "priority": 4,
                "initiative": "Monitor employee satisfaction and work-life indicators",
                "rationale": (
                    "Track satisfaction and work-life measures alongside attrition "
                    "rather than assuming they are causal drivers."
                ),
                "action": (
                    "Combine pulse-survey results with workload and manager metrics "
                    "to identify areas for further investigation."
                ),
                "metrics": [
                    "Job satisfaction",
                    "Work-life balance",
                    "Attrition rate by satisfaction segment",
                ],
                "evidence_note": "Observed associations should be validated with organizational context.",
            },
            {
                "priority": 5,
                "initiative": "Use model predictions as decision support, not decisions",
                "rationale": (
                    "The Random Forest can identify patterns associated with higher "
                    "predicted attrition probability on held-out data."
                ),
                "action": (
                    "Validate model performance and fairness before any operational "
                    "use; use predictions to prompt supportive review, never as an "
                    "automatic employment decision."
                ),
                "metrics": [
                    "Recall",
                    "PR-AUC",
                    "Calibration",
                    "Performance across relevant workforce groups",
                ],
                "evidence_note": "Predictions are probabilistic and not guarantees of employee behavior.",
            },
        ]

    def get_executive_summary(self, impact: Dict[str, Any]) -> str:
        """Return a concise business summary without unsupported savings claims."""
        total = impact.get("total_replacement_cost", 0.0)
        scenarios = impact.get("retention_scenarios", [])

        lines = [
            "EXECUTIVE SUMMARY: HR ATTRITION AND RETENTION COST ANALYSIS",
            "=" * 60,
            "",
            f"Observed replacement-cost exposure: {total:,.0f} USD",
            (
                f"Assumption: annual salary × {impact.get('replacement_cost_multiplier', self.replacement_cost_multiplier):.2f} "
                "replacement-cost multiplier."
            ),
            "",
            "Illustrative retention scenarios:",
        ]

        for scenario in scenarios:
            lines.append(
                f"- {scenario['retention_effectiveness']:.0%} effectiveness: "
                f"{scenario['avoided_cost_estimate']:,.0f} USD avoided-cost estimate"
            )

        lines.extend([
            "",
            "Important: scenario values are hypothetical estimates, not observed "
            "or guaranteed savings and not causal estimates.",
        ])
        return "\\n".join(lines)
