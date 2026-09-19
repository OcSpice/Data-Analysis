"""
Unit Tests for Financial Reconciliation Pipeline
================================================

Portfolio Category: Data Analysis

This module contains pytest tests for validating core data transformation
and masking functions in the financial reconciliation pipeline.

Tests cover:
- Data loading and validation
- Data quality engine functionality
- PII masking operations
- Anomaly detection methods
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data_loader import DataLoader
from data_quality_engine import DataQualityEngine
from pii_masker import PIIMasker
from anomaly_detector import AnomalyDetector


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_transaction_data():
    """
    Create a sample transaction DataFrame for testing.
    
    Returns:
        pd.DataFrame: Sample transaction data with known characteristics.
    """
    data = {
        "Transaction_ID": ["TXN-000001", "TXN-000002", "TXN-000003", "TXN-000004", "TXN-000005"],
        "Date": pd.to_datetime(["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "2024-05-25"]),
        "Year": [2024, 2024, 2024, 2024, 2024],
        "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
        "Quarter": ["Q1", "Q1", "Q1", "Q2", "Q2"],
        "Department": ["Finance", "HR", "IT", "Finance", "HR"],
        "Category": ["Operating Expense", "Payroll", "Equipment", "Revenue", "Payroll"],
        "Transaction_Type": ["Debit", "Debit", "Debit", "Credit", "Debit"],
        "Source_System": ["ERP System", "Bank Statement", "ERP System", "Bank Statement", "ERP System"],
        "Region": ["North America", "Europe", "Asia Pacific", "North America", "Europe"],
        "Currency": ["USD", "EUR", "USD", "USD", "EUR"],
        "Analyst": ["John Smith", "Jane Doe", "Bob Wilson", "Alice Brown", "Charlie Davis"],
        "Expected_Amount": [10000.00, 5000.00, 15000.00, 20000.00, 8000.00],
        "Actual_Amount": [10050.00, 5000.00, 14500.00, 19800.00, 8100.00],
        "Discrepancy": [50.00, 0.00, -500.00, -200.00, 100.00],
        "Discrepancy_Pct": [0.5, 0.0, -3.33, -1.0, 1.25],
        "Status": ["Reconciled", "Reconciled", "Under Review", "Under Review", "Reconciled"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_issues():
    """
    Create sample data with intentional quality issues for testing.
    
    Returns:
        pd.DataFrame: Sample data with missing values and duplicates.
    """
    data = {
        "Transaction_ID": ["TXN-000001", "TXN-000001", "TXN-000003", None, "TXN-000005"],
        "Date": pd.to_datetime(["2024-01-15", "2024-01-15", "2024-03-10", "2024-04-05", None]),
        "Department": ["Finance", "Finance", None, "IT", "HR"],
        "Expected_Amount": [10000.00, 10000.00, 15000.00, None, 8000.00],
        "Actual_Amount": [10050.00, 10050.00, 14500.00, 12000.00, 8100.00],
        "Discrepancy": [50.00, 50.00, -500.00, None, 100.00],
        "Discrepancy_Pct": [0.5, 0.5, -3.33, None, 1.25],
        "Status": ["Reconciled", "Reconciled", "Under Review", "Pending", "Reconciled"]
    }
    return pd.DataFrame(data)


# =============================================================================
# Data Loader Tests
# =============================================================================

class TestDataLoader:
    """Tests for the DataLoader class."""
    
    def test_load_data_from_csv(self, tmp_path):
        """Test loading data from a CSV file."""
        # Create a temporary CSV file
        csv_file = tmp_path / "test_data.csv"
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        df.to_csv(csv_file, index=False)
        
        # Load using DataLoader
        loader = DataLoader(str(csv_file))
        loaded_df = loader.load_data()
        
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["col1", "col2"]
    
    def test_load_data_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        loader = DataLoader("nonexistent_file.csv")
        with pytest.raises(FileNotFoundError):
            loader.load_data()
    
    def test_get_data_summary(self, sample_transaction_data):
        """Test data summary generation."""
        loader = DataLoader.__new__(DataLoader)  # Create without init
        summary = loader.get_data_summary(sample_transaction_data)
        
        assert summary["row_count"] == 5
        assert summary["column_count"] == len(sample_transaction_data.columns)
        assert "Transaction_ID" in summary["columns"]


# =============================================================================
# Data Quality Engine Tests
# =============================================================================

class TestDataQualityEngine:
    """Tests for the DataQualityEngine class."""
    
    def test_schema_validation_pass(self, sample_transaction_data):
        """Test schema validation with valid data."""
        dqe = DataQualityEngine(sample_transaction_data)
        result = dqe.validate_schema()
        
        assert result["schema_valid"] is True
        assert len(result["missing_columns"]) == 0
    
    def test_schema_validation_missing_columns(self, sample_data_with_issues):
        """Test schema validation detects missing columns."""
        dqe = DataQualityEngine(sample_data_with_issues)
        result = dqe.validate_schema()
        
        # Should detect missing columns
        assert len(result["missing_columns"]) > 0
    
    def test_missing_value_detection(self, sample_data_with_issues):
        """Test missing value detection."""
        dqe = DataQualityEngine(sample_data_with_issues)
        result = dqe.check_missing_values()
        
        assert result["total_missing"] > 0
        assert "missing_by_column" in result
    
    def test_duplicate_detection(self, sample_data_with_issues):
        """Test duplicate record detection."""
        dqe = DataQualityEngine(sample_data_with_issues)
        result = dqe.identify_duplicates()
        
        assert result["has_duplicates"] == True  # Use == instead of is for numpy bool
        assert result["duplicate_count"] > 0
    
    def test_handle_missing_values_drop(self, sample_data_with_issues):
        """Test handling missing values by dropping rows."""
        dqe = DataQualityEngine(sample_data_with_issues)
        cleaned = dqe.handle_missing_values(strategy="drop_critical")
        
        # Should have fewer rows after dropping
        assert len(cleaned) < len(sample_data_with_issues)
    
    def test_remove_duplicates(self, sample_data_with_issues):
        """Test duplicate removal."""
        dqe = DataQualityEngine(sample_data_with_issues)
        cleaned = dqe.remove_duplicates(subset=["Transaction_ID"])
        
        # Transaction_ID TXN-000001 appears twice, should be reduced
        assert len(cleaned) < len(sample_data_with_issues)
    
    def test_quality_score_calculation(self, sample_transaction_data):
        """Test quality score calculation."""
        dqe = DataQualityEngine(sample_transaction_data)
        score = dqe.calculate_quality_score()
        
        assert 0 <= score <= 1
        # Clean data should have high score
        assert score >= 0.8
    
    def test_full_validation_report(self, sample_transaction_data):
        """Test comprehensive validation report."""
        dqe = DataQualityEngine(sample_transaction_data)
        report = dqe.run_full_validation()
        
        assert "schema_valid" in report
        assert "total_missing" in report
        assert "duplicate_count" in report
        assert "quality_score" in report


# =============================================================================
# PII Masker Tests
# =============================================================================

class TestPIIMasker:
    """Tests for the PIIMasker class."""
    
    def test_mask_name_partial(self):
        """Test partial name masking."""
        masker = PIIMasker(pd.DataFrame())
        
        # Test full name masking
        masked = masker.mask_name("John Smith")
        assert masked[0] == "J"
        assert "*" in masked
        
        # Test single name
        masked_single = masker.mask_name("John")
        assert masked_single[0] == "J"
    
    def test_mask_transaction_id(self):
        """Test transaction ID masking."""
        masker = PIIMasker(pd.DataFrame())
        
        masked = masker.mask_transaction_id("TXN-000123", show_last=4)
        assert masked.startswith("TXN-")
        assert masked.endswith("0123")
        assert "*" in masked
    
    def test_hash_value_deterministic(self):
        """Test that hashing is deterministic."""
        masker = PIIMasker(pd.DataFrame())
        
        hash1 = masker.hash_value("test_value", salt="test_salt")
        hash2 = masker.hash_value("test_value", salt="test_salt")
        
        assert hash1 == hash2
    
    def test_mask_analyst_names(self, sample_transaction_data):
        """Test analyst name masking."""
        masker = PIIMasker(sample_transaction_data.copy())
        masked_df = masker.mask_analyst_names(strategy="partial")
        
        # Names should be masked (contain asterisks)
        assert "*" in masked_df["Analyst"].iloc[0]
        # First letter should be preserved
        assert masked_df["Analyst"].iloc[0][0] == sample_transaction_data["Analyst"].iloc[0][0]
    
    def test_mask_transaction_ids_in_dataframe(self, sample_transaction_data):
        """Test transaction ID masking in DataFrame."""
        masker = PIIMasker(sample_transaction_data.copy())
        masked_df = masker.mask_transaction_ids(show_last=4)
        
        # IDs should still start with TXN-
        assert masked_df["Transaction_ID"].iloc[0].startswith("TXN-")
        # Should contain asterisks
        assert "*" in masked_df["Transaction_ID"].iloc[0]
    
    def test_mask_all_pii(self, sample_transaction_data):
        """Test masking all PII columns."""
        masker = PIIMasker(sample_transaction_data.copy())
        masked_df = masker.mask_all_pii()
        
        # Both Analyst and Transaction_ID should be masked
        assert "*" in masked_df["Analyst"].iloc[0]
        assert "*" in masked_df["Transaction_ID"].iloc[0]
    
    def test_masking_report(self, sample_transaction_data):
        """Test masking report generation."""
        masker = PIIMasker(sample_transaction_data.copy())
        masker.mask_all_pii()
        report = masker.get_masking_report()
        
        assert "masked_columns" in report
        assert "masking_details" in report
        assert report["total_columns_masked"] >= 2


# =============================================================================
# Anomaly Detector Tests
# =============================================================================

class TestAnomalyDetector:
    """Tests for the AnomalyDetector class."""
    
    def test_zscore_anomaly_detection(self, sample_transaction_data):
        """Test Z-score based anomaly detection."""
        detector = AnomalyDetector(sample_transaction_data)
        anomalies = detector.detect_zscore_anomalies(threshold=2.0)
        
        # Result should be a DataFrame
        assert isinstance(anomalies, pd.DataFrame)
        # Should have zscore column added
        assert "zscore_Discrepancy_Pct" in detector.df.columns or "zscore_Discrepancy_Pct" in anomalies.columns
    
    def test_isolation_forest_detection(self, sample_transaction_data):
        """Test Isolation Forest anomaly detection."""
        detector = AnomalyDetector(sample_transaction_data)
        anomalies = detector.detect_isolation_forest_anomalies(contamination=0.2)
        
        # Result should be a DataFrame
        assert isinstance(anomalies, pd.DataFrame)
        # Model should be stored
        assert "isolation_forest" in detector.models
    
    def test_audit_priority_tiering(self, sample_transaction_data):
        """Test audit priority tiering."""
        detector = AnomalyDetector(sample_transaction_data)
        tiering = detector.create_audit_priority_tiering()
        
        assert "high_priority" in tiering
        assert "medium_priority" in tiering
        assert "low_priority" in tiering
    
    def test_combined_anomaly_score(self, sample_transaction_data):
        """Test combined anomaly score calculation."""
        detector = AnomalyDetector(sample_transaction_data)
        result_df = detector.calculate_combined_anomaly_score()
        
        assert "combined_anomaly_score" in result_df.columns
    
    def test_anomalies_by_department(self, sample_transaction_data):
        """Test department-level anomaly aggregation."""
        detector = AnomalyDetector(sample_transaction_data)
        dept_summary = detector.get_anomalies_by_department()
        
        assert "Department" in dept_summary.columns
        assert len(dept_summary) > 0
    
    def test_financial_exposure_summary(self, sample_transaction_data):
        """Test financial exposure calculation."""
        detector = AnomalyDetector(sample_transaction_data)
        exposure = detector.get_financial_exposure_summary()
        
        assert "total_discrepancy_absolute" in exposure
        assert "by_priority" in exposure
        assert "top_exposure_transactions" in exposure


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_full_pipeline_flow(self, sample_transaction_data):
        """Test the complete data processing flow."""
        # Step 1: Data Quality
        dqe = DataQualityEngine(sample_transaction_data)
        quality_report = dqe.run_full_validation()
        cleaned_df = dqe.get_cleaned_data()
        
        # Step 2: PII Masking
        masker = PIIMasker(cleaned_df)
        masked_df = masker.mask_all_pii()
        
        # Step 3: Anomaly Detection
        detector = AnomalyDetector(masked_df)
        detector.detect_zscore_anomalies()
        detector.detect_isolation_forest_anomalies()
        tiering = detector.create_audit_priority_tiering()
        
        # Verify all steps completed
        assert quality_report["quality_score"] >= 0
        assert "*" in masked_df["Analyst"].iloc[0]
        assert "audit_priority" in masked_df.columns or "audit_priority" in detector.audit_tiering.columns
    
    def test_end_to_end_with_sample_file(self, tmp_path):
        """Test end-to-end pipeline with a temporary file."""
        # Create test CSV
        csv_file = tmp_path / "test_transactions.csv"
        df = pd.DataFrame({
            "Transaction_ID": ["TXN-001", "TXN-002", "TXN-003"],
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "Year": [2024, 2024, 2024],
            "Month": ["Jan", "Jan", "Jan"],
            "Quarter": ["Q1", "Q1", "Q1"],
            "Department": ["Finance", "HR", "IT"],
            "Category": ["Expense", "Payroll", "Equipment"],
            "Transaction_Type": ["Debit", "Debit", "Debit"],
            "Source_System": ["ERP", "Bank", "ERP"],
            "Region": ["US", "EU", "US"],
            "Currency": ["USD", "EUR", "USD"],
            "Analyst": ["John Smith", "Jane Doe", "Bob Wilson"],
            "Expected_Amount": [1000, 2000, 3000],
            "Actual_Amount": [1050, 1900, 3500],
            "Discrepancy": [50, -100, 500],
            "Discrepancy_Pct": [5.0, -5.0, 16.67],
            "Status": ["Reconciled", "Under Review", "Flagged"]
        })
        df.to_csv(csv_file, index=False)
        
        # Run pipeline
        loader = DataLoader(str(csv_file))
        data = loader.load_data()
        
        dqe = DataQualityEngine(data)
        report = dqe.run_full_validation()
        
        assert report["schema_valid"] is True
        assert report["record_count"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
