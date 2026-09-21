"""
Predictive HR attrition modeling with leakage-safe preprocessing and SHAP.

The module compares Logistic Regression and Random Forest models using a
train/test split and a scikit-learn preprocessing pipeline. SHAP is applied
only to the fitted Random Forest on the held-out test set.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class AttritionModel:
    """Train, evaluate and explain employee attrition models."""

    NUMERICAL_FEATURES = [
        "Age", "DailyRate", "DistanceFromHome", "HourlyRate",
        "JobInvolvement", "JobLevel", "JobSatisfaction",
        "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
        "PercentSalaryHike", "PerformanceRating", "TotalWorkingYears",
        "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
        "YearsWithCurrManager", "ReplacementCost", "AnnualIncome",
    ]

    CATEGORICAL_FEATURES = [
        "BusinessTravel", "Department", "EducationField", "Gender",
        "JobRole", "MaritalStatus", "OverTime",
    ]

    # IsOverTime is intentionally excluded because it duplicates OverTime.
    # Keeping both would duplicate the same signal and split model explanation importance.
    ENGINEERED_FEATURES = [
        "PromotionStagnation", "LowSatisfactionCount", "PoorWorkLifeBalance",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.preprocessors: Dict[str, ColumnTransformer] = {}
        self.model = None
        self.feature_names: List[str] = []
        self.transformed_feature_names: List[str] = []
        self.X_test_raw: pd.DataFrame | None = None
        self.y_test: pd.Series | None = None
        self.X_test_transformed = None
        self.shap_values = None

    def _get_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        numeric = [
            f for f in self.NUMERICAL_FEATURES + self.ENGINEERED_FEATURES
            if f in df.columns
        ]
        categorical = [f for f in self.CATEGORICAL_FEATURES if f in df.columns]
        return numeric, categorical

    def _build_preprocessor(
        self, numeric: List[str], categorical: List[str], scale_numeric: bool
    ) -> ColumnTransformer:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]

        return ColumnTransformer(
            transformers=[
                ("num", Pipeline(numeric_steps), numeric),
                ("cat", Pipeline(categorical_steps), categorical),
            ],
            remainder="drop",
        )

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Return raw selected features and a binary target for inspection/testing."""
        if "Attrition" not in df.columns:
            raise ValueError("Column 'Attrition' not found in DataFrame")

        numeric, categorical = self._get_features(df)
        self.feature_names = numeric + categorical
        X = df[self.feature_names].copy()
        y = (df["Attrition"].astype(str).str.strip().str.lower() == "yes").astype(int)
        return X, y

    def train(self, df: pd.DataFrame, test_size: float = 0.20) -> Dict[str, Any]:
        """Train baseline Logistic Regression and Random Forest models."""
        X, y = self.prepare_features(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

        numeric, categorical = self._get_features(df)
        logistic_preprocessor = self._build_preprocessor(
            numeric, categorical, scale_numeric=True
        )
        forest_preprocessor = self._build_preprocessor(
            numeric, categorical, scale_numeric=False
        )

        models = {
            "logistic_regression": Pipeline([
                ("preprocessor", logistic_preprocessor),
                ("model", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=self.random_state,
                )),
            ]),
            "random_forest": Pipeline([
                ("preprocessor", forest_preprocessor),
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=-1,
                )),
            ]),
        }

        results: Dict[str, Any] = {"models": {}, "test_size": test_size}

        for name, pipeline in models.items():
            pipeline.fit(X_train, y_train)
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= 0.50).astype(int)

            results["models"][name] = self._evaluate(
                y_test, predictions, probabilities
            )
            self.preprocessors[name] = pipeline.named_steps["preprocessor"]

        self.models = models
        self.model = models["random_forest"]
        self.X_test_raw = X_test
        self.y_test = y_test
        self.X_test_transformed = self.preprocessors["random_forest"].transform(X_test)
        self.transformed_feature_names = list(
            self.preprocessors["random_forest"].get_feature_names_out()
        )

        rf_results = results["models"]["random_forest"]
        rf_pipeline = self.models["random_forest"]
        rf_probabilities = rf_pipeline.predict_proba(X_test)[:, 1]
        rf_predictions = (rf_probabilities >= 0.50).astype(int)
        results.update({
            "accuracy": rf_results["accuracy"],
            "auc_roc": rf_results["roc_auc"],
            "pr_auc": rf_results["pr_auc"],
            "precision": rf_results["precision"],
            "recall": rf_results["recall"],
            "f1": rf_results["f1"],
            "classification_report": rf_results["classification_report"],
            "confusion_matrix": rf_results["confusion_matrix"],
            "feature_importances": self._get_feature_importances(),
            "test_predicted_probabilities": rf_probabilities.tolist(),
            "test_predicted_labels": rf_predictions.tolist(),
            "test_actual_labels": y_test.tolist(),
        })
        return results

    @staticmethod
    def _evaluate(
        y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray
    ) -> Dict[str, Any]:
        return {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "pr_auc": float(average_precision_score(y_true, probabilities)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
            "classification_report": classification_report(
                y_true, predictions, zero_division=0
            ),
        }

    def _source_feature(self, transformed_name: str) -> str:
        """Map a transformed feature back to its original source column."""
        name = transformed_name.split("__", 1)[-1]
        for category in self.CATEGORICAL_FEATURES:
            if name.startswith(category + "_"):
                return category
        return name

    def _get_feature_importances(self) -> Dict[str, float]:
        """Return Random Forest importance aggregated to original features."""
        forest = self.models["random_forest"].named_steps["model"]
        aggregated: Dict[str, float] = {}
        for name, value in zip(
            self.transformed_feature_names, forest.feature_importances_
        ):
            source = self._source_feature(name)
            aggregated[source] = aggregated.get(source, 0.0) + float(value)
        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))

    def calculate_shap_values(self, max_samples: int = 500) -> np.ndarray:
        """Calculate genuine TreeSHAP values for held-out Random Forest data."""
        if self.model is None or self.X_test_transformed is None:
            raise ValueError("Train the model before calculating SHAP values")

        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "The 'shap' package is required for model explainability. "
                "Install dependencies from requirements.txt."
            ) from exc

        X_shap = self.X_test_transformed[:max_samples]
        explainer = shap.TreeExplainer(self.model.named_steps["model"])
        raw_values = explainer.shap_values(X_shap)

        if isinstance(raw_values, list):
            values = raw_values[1]
        else:
            values = np.asarray(raw_values)
            if values.ndim == 3:
                values = values[:, :, 1]

        self.shap_values = np.asarray(values)
        return self.shap_values

    def get_shap_summary(self, max_samples: int = 500) -> Dict[str, float]:
        """Return mean absolute TreeSHAP importance grouped by source feature."""
        if self.shap_values is None:
            self.calculate_shap_values(max_samples=max_samples)

        mean_abs = np.mean(np.abs(self.shap_values), axis=0)
        summary: Dict[str, float] = {}
        for name, value in zip(self.transformed_feature_names, mean_abs):
            source = self._source_feature(name)
            summary[source] = summary.get(source, 0.0) + float(value)
        return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))

    def explain_prediction(self, sample_idx: int = 0) -> Dict[str, Any]:
        """Explain one held-out test observation using TreeSHAP values."""
        if self.shap_values is None:
            self.calculate_shap_values()

        if self.X_test_raw is None or self.y_test is None:
            raise ValueError("Test data is not available")
        if sample_idx < 0 or sample_idx >= len(self.X_test_raw):
            raise IndexError("sample_idx is outside the held-out test set")

        row_values = self.shap_values[sample_idx]
        grouped: Dict[str, float] = {}
        for name, value in zip(self.transformed_feature_names, row_values):
            source = self._source_feature(name)
            grouped[source] = grouped.get(source, 0.0) + float(value)

        positive = sorted(
            [(k, v) for k, v in grouped.items() if v > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        negative = sorted(
            [(k, v) for k, v in grouped.items() if v < 0],
            key=lambda x: x[1],
        )

        probability = float(
            self.model.predict_proba(self.X_test_raw.iloc[[sample_idx]])[0, 1]
        )
        return {
            "sample_index": int(sample_idx),
            "predicted_probability": probability,
            "top_attrition_contributors": [
                {"feature": k, "shap_value": v} for k, v in positive[:5]
            ],
            "top_retention_contributors": [
                {"feature": k, "shap_value": v} for k, v in negative[:5]
            ],
        }

    def train_and_explain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train, evaluate and explain the Random Forest model."""
        results = self.train(df)
        shap_summary = self.get_shap_summary()
        results["shap_summary"] = shap_summary
        results["top_features"] = dict(list(shap_summary.items())[:10])
        results["example_explanation"] = self.explain_prediction(0)
        return results
