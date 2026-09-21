"""
Unit Tests for HR Attrition Analysis Pipeline

Tests core data transformation, cost calculation, and PII masking functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import DataLoader
from data_validator import DataValidator
from pii_masker import PIIMasker
from feature_engineer import FeatureEngineer


class TestDataLoader:
    """Tests for the DataLoader class."""
    
    def test_load_csv_file(self):
        """Test that CSV file loads correctly."""
        data_path = Path(__file__).parent.parent / "HR-Employee-Attrition-Dataset.csv"
        loader = DataLoader(str(data_path))
        df = loader.load()
        
        assert len(df) > 0
        assert len(df.columns) > 0
        assert 'Attrition' in df.columns
        assert 'EmployeeNumber' in df.columns
    
    def test_get_summary(self):
        """Test summary generation."""
        data_path = Path(__file__).parent.parent / "HR-Employee-Attrition-Dataset.csv"
        loader = DataLoader(str(data_path))
        df = loader.load()
        summary = loader.get_summary()
        
        assert 'row_count' in summary
        assert 'column_count' in summary
        assert summary['row_count'] == len(df)
        assert summary['column_count'] == len(df.columns)
    
    def test_validate_schema(self):
        """Test schema validation."""
        data_path = Path(__file__).parent.parent / "HR-Employee-Attrition-Dataset.csv"
        loader = DataLoader(str(data_path))
        df = loader.load()
        is_valid, missing = loader.validate_schema()
        
        # Dataset should have all expected columns
        assert is_valid is True
        assert len(missing) == 0


class TestDataValidator:
    """Tests for the DataValidator class."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'Age': [25, 30, 35, 40],
            'Attrition': ['Yes', 'No', 'Yes', 'No'],
            'OverTime': ['Yes', 'No', 'Yes', 'No'],
            'MonthlyIncome': [3000, 5000, 7000, 9000],
            'YearsAtCompany': [1, 5, 3, 10],
            'Department': ['Sales', 'R&D', 'Sales', 'R&D'],
            'EmployeeNumber': [1, 2, 3, 4]
        })
    
    def test_validate_required_columns(self, sample_df):
        """Test validation of required columns."""
        validator = DataValidator()
        report = validator.validate(sample_df)
        
        assert 'status' in report
        assert 'missing_columns' in report
        assert 'missing_values' in report
    
    def test_handle_missing_values_median(self, sample_df):
        """Test handling missing values with median strategy."""
        # Add some missing values
        sample_df.loc[0, 'Age'] = np.nan
        sample_df.loc[1, 'MonthlyIncome'] = np.nan
        
        validator = DataValidator()
        df_clean = validator.handle_missing_values(sample_df, strategy='median')
        
        assert df_clean.isnull().sum().sum() == 0
        assert df_clean['Age'].iloc[0] == sample_df['Age'].median()
    
    def test_handle_missing_values_mode(self, sample_df):
        """Test handling missing values with mode strategy."""
        sample_df.loc[0, 'Department'] = None
        
        validator = DataValidator()
        df_clean = validator.handle_missing_values(sample_df, strategy='mode')
        
        assert df_clean.isnull().sum().sum() == 0


class TestPIIMasker:
    """Tests for the PIIMasker class."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame with PII for testing."""
        return pd.DataFrame({
            'EmployeeNumber': [1, 2, 3, 4, 5],
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'Attrition': ['Yes', 'No', 'Yes', 'No', 'Yes']
        })
    
    def test_hash_masking(self, sample_df):
        """Test hash-based PII masking."""
        masker = PIIMasker(salt='test_salt')
        df_masked = masker.mask(sample_df, columns=['EmployeeNumber'], strategy='hash')
        
        # Check that values are masked
        assert df_masked['EmployeeNumber'].iloc[0].startswith('EMP_')
        assert df_masked['EmployeeNumber'].iloc[0] != str(sample_df['EmployeeNumber'].iloc[0])
    
    def test_redact_masking(self, sample_df):
        """Test redaction-based PII masking."""
        masker = PIIMasker()
        df_masked = masker.mask(sample_df, columns=['EmployeeNumber'], strategy='redact')
        
        # All values should be redacted
        assert all(df_masked['EmployeeNumber'] == '[REDACTED]')
    
    def test_pseudonymize_masking(self, sample_df):
        """Test pseudonymization-based PII masking."""
        masker = PIIMasker()
        df_masked = masker.mask(sample_df, columns=['EmployeeNumber'], strategy='pseudonymize')
        
        # Values should be replaced with ID_X format
        assert df_masked['EmployeeNumber'].iloc[0].startswith('ID_')
        assert len(df_masked['EmployeeNumber'].unique()) == len(sample_df['EmployeeNumber'].unique())
    
    def test_verify_masking(self, sample_df):
        """Test masking verification."""
        masker = PIIMasker(salt='test_salt')
        df_masked = masker.mask(sample_df, columns=['EmployeeNumber'], strategy='hash')
        
        # Verification should pass
        assert masker.verify_masking(df_masked) is True
        
        # Verification should fail on original data
        assert masker.verify_masking(sample_df) is False
    
    def test_get_masking_report(self, sample_df):
        """Test masking report generation."""
        masker = PIIMasker(salt='test_salt')
        df_masked = masker.mask(sample_df, columns=['EmployeeNumber'])
        report = masker.get_masking_report()
        
        assert 'masked_columns' in report
        assert 'masking_strategy' in report
        assert 'timestamp' in report
        assert 'EmployeeNumber' in report['masked_columns']


class TestFeatureEngineer:
    """Tests for the FeatureEngineer class."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing feature engineering."""
        return pd.DataFrame({
            'MonthlyIncome': [3000, 5000, 7000, 9000],
            'YearsAtCompany': [1, 5, 3, 10],
            'TotalWorkingYears': [2, 8, 5, 15],
            'OverTime': ['Yes', 'No', 'Yes', 'No'],
            'NumCompaniesWorked': [3, 2, 5, 1],
            'YearsSinceLastPromotion': [0, 4, 2, 5],
            'JobLevel': [1, 3, 2, 4],
            'EnvironmentSatisfaction': [2, 4, 1, 3],
            'JobSatisfaction': [3, 4, 2, 4],
            'RelationshipSatisfaction': [2, 3, 1, 4],
            'WorkLifeBalance': [1, 3, 2, 4],
            'Attrition': ['Yes', 'No', 'Yes', 'No']
        })
    
    def test_create_financial_features(self, sample_df):
        """Test creation of financial features."""
        engineer = FeatureEngineer(replacement_cost_multiplier=1.5)
        df_enriched = engineer.create_features(sample_df)
        
        assert 'AnnualIncome' in df_enriched.columns
        assert 'ReplacementCost' in df_enriched.columns
        
        # Verify calculations
        assert df_enriched['AnnualIncome'].iloc[0] == 3000 * 12
        assert df_enriched['ReplacementCost'].iloc[0] == 3000 * 12 * 1.5
    
    def test_create_tenure_features(self, sample_df):
        """Test creation of tenure features."""
        engineer = FeatureEngineer()
        df_enriched = engineer.create_features(sample_df)
        
        assert 'TenureBucket' in df_enriched.columns
        assert 'PromotionStagnation' in df_enriched.columns
        
        # Check promotion stagnation flag
        assert df_enriched['PromotionStagnation'].iloc[1] == 1  # 4 years since promotion
        assert df_enriched['PromotionStagnation'].iloc[2] == 0  # 2 years since promotion
    
    def test_create_risk_features(self, sample_df):
        """Test creation of risk indicator features."""
        engineer = FeatureEngineer()
        df_enriched = engineer.create_features(sample_df)
        
        assert 'IsOverTime' in df_enriched.columns
        assert 'LowSatisfactionCount' in df_enriched.columns
        assert 'HasLowSatisfaction' in df_enriched.columns
        assert 'PoorWorkLifeBalance' in df_enriched.columns
        
        # Verify overtime flag
        assert df_enriched['IsOverTime'].iloc[0] == 1
        assert df_enriched['IsOverTime'].iloc[1] == 0
    
    def test_calculate_total_replacement_cost(self, sample_df):
        """Test total replacement cost calculation."""
        engineer = FeatureEngineer(replacement_cost_multiplier=1.5)
        df_enriched = engineer.create_features(sample_df)
        
        total_cost = engineer.calculate_total_replacement_cost(df_enriched)
        
        # Only employees with Attrition='Yes' should be counted
        departed = df_enriched[df_enriched['Attrition'] == 'Yes']
        expected_cost = departed['ReplacementCost'].sum()
        
        assert total_cost == expected_cost
    
    def test_get_feature_summary(self, sample_df):
        """Test feature summary generation."""
        engineer = FeatureEngineer()
        df_enriched = engineer.create_features(sample_df)
        summary = engineer.get_feature_summary(df_enriched)
        
        assert 'features_created' in summary
        assert 'statistics' in summary
        assert 'AnnualIncome' in summary['features_created']


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_data_flow(self):
        """Test the complete data processing flow."""
        # Load data
        data_path = Path(__file__).parent.parent / "HR-Employee-Attrition-Dataset.csv"
        loader = DataLoader(str(data_path))
        df = loader.load()
        
        # Validate
        validator = DataValidator()
        report = validator.validate(df)
        assert report['status'] in ['passed', 'warning']
        
        # Mask PII
        masker = PIIMasker()
        df_masked = masker.mask(df, columns=['EmployeeNumber'])
        assert masker.verify_masking(df_masked) is True
        
        # Engineer features
        engineer = FeatureEngineer()
        df_enriched = engineer.create_features(df_masked)
        assert 'ReplacementCost' in df_enriched.columns
        assert 'IsOverTime' in df_enriched.columns
        
        # Verify no PII leakage
        assert not df_enriched['EmployeeNumber'].astype(str).str.contains(r'^\d+$').any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestAttritionModel:
    """Tests for leakage-safe predictive modeling and evaluation."""

    def test_model_training_produces_required_metrics(self):
        from attrition_model import AttritionModel

        data_path = Path(__file__).parent.parent / "HR-Employee-Attrition-Dataset.csv"
        df = DataLoader(str(data_path)).load()
        df = FeatureEngineer().create_features(df)

        model = AttritionModel(random_state=42)
        results = model.train(df)

        assert set(["logistic_regression", "random_forest"]) == set(results["models"])
        for metrics in results["models"].values():
            for key in ["accuracy", "roc_auc", "pr_auc", "precision", "recall", "f1"]:
                assert 0.0 <= metrics[key] <= 1.0
        assert len(results["test_predicted_probabilities"]) == len(model.y_test)

    def test_real_shap_values_are_generated(self):
        pytest.importorskip("shap")
        from attrition_model import AttritionModel

        data_path = Path(__file__).parent.parent / "HR-Employee-Attrition-Dataset.csv"
        df = DataLoader(str(data_path)).load()
        df = FeatureEngineer().create_features(df)

        model = AttritionModel(random_state=42)
        results = model.train_and_explain(df)

        assert model.shap_values is not None
        assert model.shap_values.ndim == 2
        assert len(results["top_features"]) > 0


class TestBusinessImpact:
    """Tests for scenario-based financial modeling."""

    def test_retention_scenarios_are_bounded(self):
        from business_impact import BusinessImpactAnalyzer

        df = pd.DataFrame({
            "Attrition": ["Yes", "No", "Yes"],
            "AnnualIncome": [40000, 50000, 60000],
        })
        analyzer = BusinessImpactAnalyzer(replacement_cost_multiplier=1.5)
        impact = analyzer.calculate_impact(df, {}, {})

        assert impact["total_replacement_cost"] == (40000 + 60000) * 1.5
        assert len(impact["retention_scenarios"]) == 5
        assert impact["retention_scenarios"][0]["avoided_cost_estimate"] == impact["total_replacement_cost"] * 0.10
        assert impact["retention_scenarios"][-1]["avoided_cost_estimate"] == impact["total_replacement_cost"] * 0.50

    def test_invalid_scenario_effectiveness_is_rejected(self):
        from business_impact import BusinessImpactAnalyzer

        with pytest.raises(ValueError):
            BusinessImpactAnalyzer(scenario_effectiveness=[-0.1, 0.2])

        with pytest.raises(ValueError):
            BusinessImpactAnalyzer(scenario_effectiveness=[0.2, 1.1])
