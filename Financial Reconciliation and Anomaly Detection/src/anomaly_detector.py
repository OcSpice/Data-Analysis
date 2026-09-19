"""
Anomaly Detection Engine Module
===============================

Implements advanced statistical and machine learning techniques for detecting
anomalous financial transactions, quantifying financial exposure, and creating
audit priority tiers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """
    Advanced anomaly detection engine for financial transaction data.
    
    This class implements multiple anomaly detection approaches:
    - Z-score based statistical detection
    - Isolation Forest (unsupervised ML)
    - Combined ensemble scoring
    - Audit priority tiering
    
    Attributes:
        df (pd.DataFrame): Transaction DataFrame to analyze.
        anomaly_results (Dict): Storage for detection results.
        models (Dict): Trained anomaly detection models.
    """
    
    # Columns used for anomaly detection features
    FEATURE_COLUMNS = [
        "Discrepancy",
        "Discrepancy_Pct",
        "Expected_Amount",
        "Actual_Amount"
    ]
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Anomaly Detector.
        
        Args:
            df: DataFrame containing transaction data with discrepancy info.
        """
        self.df = df.copy()
        self.anomaly_results: Dict = {}
        self.models: Dict = {}
        self.audit_tiering = None  # Initialize audit_tiering attribute
        self._prepare_features()
    
    def _prepare_features(self):
        """
        Prepare and scale features for anomaly detection.
        
        Creates a feature matrix from relevant numeric columns and
        applies standard scaling for ML algorithms.
        """
        # Select available feature columns
        available_features = [col for col in self.FEATURE_COLUMNS if col in self.df.columns]
        
        if len(available_features) == 0:
            raise ValueError("No feature columns found for anomaly detection")
        
        # Extract features and handle missing values
        self.feature_matrix = self.df[available_features].copy()
        self.feature_matrix = self.feature_matrix.fillna(0)
        
        # Scale features for ML algorithms
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.feature_matrix)
        
        self.available_features = available_features
    
    def detect_zscore_anomalies(
        self,
        threshold: float = 3.0,
        column: str = "Discrepancy_Pct"
    ) -> pd.DataFrame:
        """
        Detect anomalies using Z-score statistical method.
        
        The Z-score measures how many standard deviations a data point
        is from the mean. Values beyond the threshold are flagged as anomalies.
        
        Args:
            threshold: Z-score threshold for anomaly flagging (default: 3.0).
            column: Column to analyze for anomalies.
            
        Returns:
            pd.DataFrame: DataFrame with anomaly flags and z-scores.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        # Calculate Z-scores
        mean_val = self.df[column].mean()
        std_val = self.df[column].std()
        
        # Avoid division by zero
        if std_val == 0:
            z_scores = pd.Series(0, index=self.df.index)
        else:
            z_scores = (self.df[column] - mean_val) / std_val
        
        # Flag anomalies
        anomaly_flags = np.abs(z_scores) > threshold
        
        # Store results directly in self.df
        self.df[f"zscore_{column}"] = z_scores
        self.df[f"is_zscore_anomaly"] = anomaly_flags
        
        self.anomaly_results["zscore"] = {
            "method": "zscore",
            "column": column,
            "threshold": threshold,
            "anomaly_count": int(anomaly_flags.sum()),
            "anomaly_percentage": float(anomaly_flags.mean() * 100),
            "timestamp": datetime.now().isoformat()
        }
        
        return self.df[anomaly_flags].copy()
    
    def detect_isolation_forest_anomalies(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest algorithm.
        
        Isolation Forest is an unsupervised ML algorithm that isolates
        anomalies by randomly selecting a feature and then randomly
        selecting a split value between the maximum and minimum values
        of the selected feature.
        
        Args:
            contamination: Expected proportion of outliers in the dataset.
            n_estimators: Number of base estimators in the ensemble.
            random_state: Random seed for reproducibility.
            
        Returns:
            pd.DataFrame: DataFrame rows flagged as anomalies.
        """
        # Train Isolation Forest model
        iso_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Fit and predict
        predictions = iso_forest.fit_predict(self.scaled_features)
        
        # Isolation Forest returns -1 for anomalies, 1 for normal
        anomaly_flags = predictions == -1
        
        # Get anomaly scores (lower = more anomalous)
        anomaly_scores = iso_forest.score_samples(self.scaled_features)
        
        # Store results directly in self.df
        self.df["isolation_forest_flag"] = anomaly_flags
        self.df["isolation_forest_score"] = anomaly_scores
        
        # Store model
        self.models["isolation_forest"] = iso_forest
        
        self.anomaly_results["isolation_forest"] = {
            "method": "isolation_forest",
            "contamination": contamination,
            "n_estimators": n_estimators,
            "anomaly_count": int(anomaly_flags.sum()),
            "anomaly_percentage": float(anomaly_flags.mean() * 100),
            "features_used": self.available_features,
            "timestamp": datetime.now().isoformat()
        }
        
        return self.df[anomaly_flags].copy()
    
    def calculate_combined_anomaly_score(self) -> pd.DataFrame:
        """
        Calculate a combined anomaly score from multiple methods.
        
        Combines Z-score and Isolation Forest results into a unified
        anomaly score for each transaction.
        
        Returns:
            pd.DataFrame: DataFrame with combined anomaly scores.
        """
        result_df = self.df.copy()
        
        # Ensure both methods have been run
        if "is_zscore_anomaly" not in result_df.columns:
            self.detect_zscore_anomalies()
        
        if "isolation_forest_flag" not in result_df.columns:
            self.detect_isolation_forest_anomalies()
        
        # Normalize Z-scores to 0-1 range
        zscore_col = [c for c in result_df.columns if c.startswith("zscore_")]
        if zscore_col:
            zscore_abs = np.abs(result_df[zscore_col[0]])
            zscore_normalized = zscore_abs / (zscore_abs.max() + 1e-10)
            result_df["zscore_normalized"] = zscore_normalized
        
        # Normalize Isolation Forest scores to 0-1 range
        if "isolation_forest_score" in result_df.columns:
            if_scores = result_df["isolation_forest_score"]
            if_normalized = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
            # Invert so higher = more anomalous
            result_df["if_normalized"] = 1 - if_normalized
        
        # Calculate combined score (weighted average)
        score_columns = []
        if "zscore_normalized" in result_df.columns:
            score_columns.append("zscore_normalized")
        if "if_normalized" in result_df.columns:
            score_columns.append("if_normalized")
        
        if score_columns:
            result_df["combined_anomaly_score"] = result_df[score_columns].mean(axis=1)
        else:
            result_df["combined_anomaly_score"] = 0
        
        self.anomaly_results["combined_score"] = {
            "method": "weighted_average",
            "components": score_columns,
            "high_score_count": int((result_df["combined_anomaly_score"] > 0.7).sum()),
            "timestamp": datetime.now().isoformat()
        }
        
        return result_df
    
    def create_audit_priority_tiering(self) -> Dict:
        """
        Create a tiered audit priority system based on anomaly detection.
        
        Tiers:
        - HIGH: Flagged by both methods OR discrepancy > 10%
        - MEDIUM: Flagged by one method OR discrepancy 5-10%
        - LOW: Minor anomalies or discrepancy 2-5%
        
        Returns:
            dict: Summary of audit priorities by tier.
        """
        result_df = self.calculate_combined_anomaly_score()
        
        # Define priority tiers
        conditions = [
            # HIGH priority: Both methods flag OR high discrepancy
            (
                (result_df.get("is_zscore_anomaly", False) & 
                 result_df.get("isolation_forest_flag", False)) |
                (result_df["Discrepancy_Pct"].abs() >= 10)
            ),
            # MEDIUM priority: One method flags OR medium discrepancy
            (
                (result_df.get("is_zscore_anomaly", False) ^ 
                 result_df.get("isolation_forest_flag", False)) |
                ((result_df["Discrepancy_Pct"].abs() >= 5) & 
                 (result_df["Discrepancy_Pct"].abs() < 10))
            ),
            # LOW priority: Lower discrepancy but still notable
            (
                (result_df["Discrepancy_Pct"].abs() >= 2) & 
                (result_df["Discrepancy_Pct"].abs() < 5)
            )
        ]
        
        priorities = ["HIGH", "MEDIUM", "LOW"]
        
        # Apply tiering
        result_df["audit_priority"] = np.select(conditions, priorities, default="NORMAL")
        
        # Calculate financial exposure by tier
        tier_summary = {}
        for tier in ["HIGH", "MEDIUM", "LOW", "NORMAL"]:
            tier_data = result_df[result_df["audit_priority"] == tier]
            tier_summary[tier.lower()] = {
                "count": len(tier_data),
                "total_discrepancy": float(tier_data["Discrepancy"].abs().sum()) if "Discrepancy" in tier_data.columns else 0,
                "avg_discrepancy_pct": float(tier_data["Discrepancy_Pct"].abs().mean()) if "Discrepancy_Pct" in tier_data.columns else 0
            }
        
        self.audit_tiering = result_df
        self.anomaly_results["audit_tiering"] = {
            "tier_summary": tier_summary,
            "total_flagged": len(result_df[result_df["audit_priority"] != "NORMAL"]),
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "high_priority": tier_summary["high"]["count"],
            "medium_priority": tier_summary["medium"]["count"],
            "low_priority": tier_summary["low"]["count"],
            "normal": tier_summary["normal"]["count"],
            "financial_exposure_high": tier_summary["high"]["total_discrepancy"],
            "financial_exposure_medium": tier_summary["medium"]["total_discrepancy"],
            "financial_exposure_low": tier_summary["low"]["total_discrepancy"]
        }
    
    def get_anomalies_by_department(self) -> pd.DataFrame:
        """
        Aggregate anomaly counts by department.
        
        Returns:
            pd.DataFrame: Anomaly statistics grouped by department.
        """
        if self.audit_tiering is None:
            self.create_audit_priority_tiering()
        
        dept_summary = self.audit_tiering.groupby("Department").agg({
            "Transaction_ID": "count",
            "Discrepancy": ["sum", "mean", "std"],
            "audit_priority": lambda x: (x == "HIGH").sum()
        }).reset_index()
        
        dept_summary.columns = [
            "Department",
            "Total_Transactions",
            "Total_Discrepancy",
            "Avg_Discrepancy",
            "Std_Discrepancy",
            "High_Priority_Count"
        ]
        
        dept_summary["High_Priority_Rate"] = (
            dept_summary["High_Priority_Count"] / dept_summary["Total_Transactions"] * 100
        )
        
        return dept_summary.sort_values("High_Priority_Count", ascending=False)
    
    def get_anomalies_by_source_system(self) -> pd.DataFrame:
        """
        Aggregate anomaly counts by source system.
        
        Returns:
            pd.DataFrame: Anomaly statistics grouped by source system.
        """
        if self.audit_tiering is None:
            self.create_audit_priority_tiering()
        
        system_summary = self.audit_tiering.groupby("Source_System").agg({
            "Transaction_ID": "count",
            "Discrepancy": ["sum", "mean"],
            "audit_priority": lambda x: (x.isin(["HIGH", "MEDIUM"])).sum()
        }).reset_index()
        
        system_summary.columns = [
            "Source_System",
            "Total_Transactions",
            "Total_Discrepancy",
            "Avg_Discrepancy",
            "Flagged_Count"
        ]
        
        system_summary["Flagged_Rate"] = (
            system_summary["Flagged_Count"] / system_summary["Total_Transactions"] * 100
        )
        
        return system_summary.sort_values("Flagged_Rate", ascending=False)
    
    def get_financial_exposure_summary(self) -> Dict:
        """
        Calculate total financial exposure from detected anomalies.
        
        Returns:
            dict: Financial exposure metrics by category and priority.
        """
        if self.audit_tiering is None:
            self.create_audit_priority_tiering()
        
        df = self.audit_tiering
        
        exposure = {
            "total_discrepancy_absolute": float(df["Discrepancy"].abs().sum()),
            "by_priority": {},
            "by_category": {},
            "top_exposure_transactions": []
        }
        
        # By priority
        for priority in ["HIGH", "MEDIUM", "LOW", "NORMAL"]:
            priority_data = df[df["audit_priority"] == priority]
            exposure["by_priority"][priority] = {
                "count": len(priority_data),
                "total_exposure": float(priority_data["Discrepancy"].abs().sum()),
                "avg_exposure": float(priority_data["Discrepancy"].abs().mean())
            }
        
        # By category
        if "Category" in df.columns:
            for category in df["Category"].unique():
                cat_data = df[df["Category"] == category]
                exposure["by_category"][category] = {
                    "count": len(cat_data),
                    "total_exposure": float(cat_data["Discrepancy"].abs().sum())
                }
        
        # Top 10 exposure transactions
        top_txns = df.nlargest(10, "Discrepancy")[
            ["Transaction_ID", "Department", "Discrepancy", "audit_priority"]
        ].to_dict("records")
        exposure["top_exposure_transactions"] = top_txns
        
        return exposure
    
    def get_all_anomaly_results(self) -> Dict:
        """
        Get comprehensive anomaly detection results.
        
        Returns:
            dict: All anomaly detection results and summaries.
        """
        return {
            "detection_methods": self.anomaly_results,
            "audit_tiering_summary": self.create_audit_priority_tiering(),
            "department_analysis": self.get_anomalies_by_department().to_dict("records"),
            "source_system_analysis": self.get_anomalies_by_source_system().to_dict("records"),
            "financial_exposure": self.get_financial_exposure_summary()
        }
