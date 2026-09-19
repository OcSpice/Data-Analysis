"""
Financial Reconciliation and Anomaly Detection Package
======================================================

Portfolio Category: Data Analysis (enhanced with Machine Learning)

This package provides a comprehensive pipeline for:
- Data quality validation and schema checking
- PII masking for compliance
- Advanced anomaly detection using statistical and ML methods
- Automated reporting and visualization

Modules:
    data_loader: Handles loading transaction data from CSV files.
    data_quality_engine: Validates data quality and handles missing values.
    pii_masker: Anonymizes sensitive fields for privacy compliance.
    anomaly_detector: Detects anomalies using Z-score and Isolation Forest.
    report_generator: Creates visualizations and summary reports.
"""

from .data_loader import DataLoader
from .data_quality_engine import DataQualityEngine
from .pii_masker import PIIMasker
from .anomaly_detector import AnomalyDetector
from .report_generator import ReportGenerator

__version__ = "1.0.0"
__author__ = "Oghenochuko Emmanuel Ogidiagba"
__all__ = [
    "DataLoader",
    "DataQualityEngine",
    "PIIMasker",
    "AnomalyDetector",
    "ReportGenerator"
]
