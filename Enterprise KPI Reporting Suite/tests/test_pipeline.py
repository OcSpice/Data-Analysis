"""
Unit Tests for Enterprise KPI Reporting Suite
Tests data transformation, KPI calculation logic, and anonymization functions.

Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.data_loader import DataLoader
from src.anonymizer import DataAnonymizer
from src.kpi_engine import KPIAnalyticsEngine
from src.visualization import ReportGenerator


# Test fixtures
@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        'Record_ID': ['REC-001', 'REC-002', 'REC-003', 'REC-004'],
        'Date': ['2024-01-15', '2024-02-15', '2024-03-15', '2024-04-15'],
        'Year': [2024, 2024, 2024, 2024],
        'Month': ['Jan', 'Feb', 'Mar', 'Apr'],
        'Quarter': ['Q1', 'Q1', 'Q1', 'Q2'],
        'Department': ['Finance', 'Sales', 'HR', 'Operations'],
        'Region': ['North America', 'Europe', 'Asia Pacific', 'Latin America'],
        'Product': ['Pro Plan', 'Enterprise Suite', 'Add-On Services', 'Pro Plan'],
        'Channel': ['Direct Sales', 'Online', 'Inbound', 'Direct Sales'],
        'Employee_ID': ['EMP-001', 'EMP-002', 'EMP-003', 'EMP-004'],
        'Cost_Center': ['CC-100', 'CC-200', 'CC-300', 'CC-400'],
        'Revenue': [10000, 15000, 8000, 12000],
        'Cost': [6000, 9000, 5000, 7000],
        'Gross_Margin': [4000, 6000, 3000, 5000],
        'Margin_Pct': [40.0, 40.0, 37.5, 41.7],
        'Deals_Closed': [5, 8, 0, 3],
        'Leads_Generated': [20, 30, 15, 25],
        'Conv_Rate_Pct': [25.0, 26.7, 0.0, 12.0],
        'Customer_Sat': [8.5, 9.0, 7.5, 8.0],
        'NPS': [70, 75, 60, 65],
        'Headcount': [150, 200, 100, 175],
        'Attrition_Pct': [10.0, 8.0, 15.0, 11.0],
        'Training_Hrs': [25, 30, 15, 20],
        'Productivity_Pct': [75.0, 80.0, 65.0, 72.0],
        'CAC': [500, 600, 400, 550],
        'LTV': [2000, 2500, 1500, 2200],
        'LTV_CAC_Ratio': [4.0, 4.17, 3.75, 4.0],
        'MQL_Count': [100, 150, 80, 120],
        'SQL_Count': [50, 75, 40, 60],
        'CTR_Pct': [3.5, 4.0, 2.8, 3.2],
        'SLA_Met': [1, 1, 0, 1],
        'Ticket_Volume': [200, 250, 150, 180],
        'Resolution_Hrs': [10.5, 8.0, 15.0, 11.0],
        'Uptime_Pct': [99.8, 99.9, 99.2, 99.5],
        'Status': ['Above Target', 'Above Target', 'Below Target', 'Above Target']
    })


@pytest.fixture
def temp_csv_path(tmp_path, sample_df):
    """Create a temporary CSV file for testing DataLoader."""
    csv_path = tmp_path / "test_data.csv"
    sample_df.to_csv(csv_path, index=False)
    return str(csv_path)


class TestDataLoader:
    """Test cases for the DataLoader class."""
    
    def test_load_csv(self, temp_csv_path):
        """Test that CSV file loads correctly."""
        loader = DataLoader(temp_csv_path)
        df = loader.load()
        
        assert len(df) == 4
        assert 'Revenue' in df.columns
        assert 'Department' in df.columns
    
    def test_validate_schema_success(self, temp_csv_path):
        """Test schema validation with correct columns."""
        loader = DataLoader(temp_csv_path)
        loader.load()
        is_valid, missing = loader.validate_schema()
        
        assert is_valid is True
        assert len(missing) == 0
    
    def test_validate_schema_failure(self, tmp_path):
        """Test schema validation with missing columns."""
        # Create CSV with missing columns
        incomplete_df = pd.DataFrame({
            'Record_ID': ['REC-001'],
            'Revenue': [1000]
            # Missing most required columns
        })
        csv_path = tmp_path / "incomplete.csv"
        incomplete_df.to_csv(csv_path, index=False)
        
        loader = DataLoader(str(csv_path))
        loader.load()
        is_valid, missing = loader.validate_schema()
        
        assert is_valid is False
        assert len(missing) > 0
    
    def test_check_data_quality(self, temp_csv_path):
        """Test data quality check functionality."""
        loader = DataLoader(temp_csv_path)
        loader.load()
        quality_report = loader.check_data_quality()
        
        assert 'total_records' in quality_report
        assert 'quality_score' in quality_report
        assert quality_report['total_records'] == 4
    
    def test_handle_missing_values_drop(self, tmp_path):
        """Test handling missing values with drop strategy."""
        df_with_na = pd.DataFrame({
            'Record_ID': ['REC-001', 'REC-002', 'REC-003'],
            'Revenue': [1000, None, 3000],
            'Cost': [500, 600, 700]
        })
        csv_path = tmp_path / "missing.csv"
        df_with_na.to_csv(csv_path, index=False)
        
        loader = DataLoader(str(csv_path))
        loader.EXPECTED_COLUMNS = ['Record_ID', 'Revenue', 'Cost']
        loader.load()
        cleaned_df = loader.handle_missing_values(strategy='drop')
        
        assert len(cleaned_df) == 2


class TestDataAnonymizer:
    """Test cases for the DataAnonymizer class."""
    
    def test_hash_employee_id(self, sample_df):
        """Test employee ID hashing."""
        anon = DataAnonymizer(sample_df)
        result_df = anon.hash_employee_id('Employee_ID')
        
        # Check that IDs are transformed
        original_ids = sample_df['Employee_ID'].tolist()
        hashed_ids = result_df['Employee_ID'].tolist()
        
        assert original_ids != hashed_ids
        assert all(len(h) == 16 for h in hashed_ids)  # SHA256 truncated to 16 chars
    
    def test_mask_employee_id(self, sample_df):
        """Test employee ID masking."""
        anon = DataAnonymizer(sample_df.copy())
        result_df = anon.mask_employee_id('Employee_ID', show_chars=3)
        
        masked_ids = result_df['Employee_ID'].tolist()
        
        # Check format: first 3 chars + asterisks
        for masked_id in masked_ids:
            assert masked_id.startswith('EMP')
            assert '*' in masked_id
    
    def test_anonymization_report(self, sample_df):
        """Test anonymization report generation."""
        anon = DataAnonymizer(sample_df)
        anon.hash_employee_id('Employee_ID')
        report = anon.get_anonymization_report()
        
        assert report['author'] == "OGHENEOCHUKO EMMANUEL OGIDIAGBA"
        assert 'Employee_ID' in report['anonymized_columns']
        assert report['privacy_compliant'] is True
    
    def test_author_constant(self):
        """Test that author constant is defined correctly."""
        assert DataAnonymizer.AUTHOR == "OGHENEOCHUKO EMMANUEL OGIDIAGBA"


class TestKPIAnalyticsEngine:
    """Test cases for the KPIAnalyticsEngine class."""
    
    def test_calculate_department_kpis(self, sample_df):
        """Test department KPI calculation."""
        engine = KPIAnalyticsEngine(sample_df)
        dept_kpis = engine.calculate_department_kpis()
        
        assert len(dept_kpis) == 4  # 4 departments
        assert 'Revenue' in dept_kpis.columns
        assert 'Margin_Pct' in dept_kpis.columns
    
    def test_calculate_regional_performance(self, sample_df):
        """Test regional performance calculation."""
        engine = KPIAnalyticsEngine(sample_df)
        regional_kpis = engine.calculate_regional_performance()
        
        assert len(regional_kpis) == 4  # 4 regions
        assert 'Revenue' in regional_kpis.columns
    
    def test_calculate_department_health_score(self, sample_df):
        """Test department health score calculation."""
        engine = KPIAnalyticsEngine(sample_df)
        
        # Test each department
        for dept in sample_df['Department'].unique():
            score = engine.calculate_department_health_score(dept)
            assert 0 <= score <= 100
    
    def test_identify_underperforming_areas(self, sample_df):
        """Test identification of underperforming areas."""
        engine = KPIAnalyticsEngine(sample_df)
        underperforming = engine.identify_underperforming_areas(top_n=3)
        
        assert isinstance(underperforming, list)
        if underperforming:
            assert 'type' in underperforming[0]
            assert 'below_target_rate' in underperforming[0]
    
    def test_get_below_target_records(self, sample_df):
        """Test filtering below target records."""
        engine = KPIAnalyticsEngine(sample_df)
        below_df = engine.get_below_target_records()
        
        assert len(below_df) == 1  # Only REC-003 is Below Target
        assert below_df['Status'].iloc[0] == 'Below Target'
    
    def test_generate_executive_summary(self, sample_df):
        """Test executive summary generation."""
        engine = KPIAnalyticsEngine(sample_df)
        summary = engine.generate_executive_summary()
        
        assert summary['author'] == "OGHENEOCHUKO EMMANUEL OGIDIAGBA"
        assert 'data_scope' in summary
        assert 'performance' in summary
        assert summary['data_scope']['total_records'] == 4
    
    def test_author_constant(self):
        """Test that author constant is defined correctly."""
        assert KPIAnalyticsEngine.AUTHOR == "OGHENEOCHUKO EMMANUEL OGIDIAGBA"


class TestReportGenerator:
    """Test cases for the ReportGenerator class."""
    
    def test_create_revenue_vs_cost_trend(self, sample_df, tmp_path):
        """Test revenue vs cost trend chart creation."""
        report_gen = ReportGenerator(sample_df, str(tmp_path))
        fig = report_gen.create_revenue_vs_cost_trend()
        
        assert fig is not None
        assert len(fig.data) == 2  # Revenue and Cost traces
    
    def test_create_department_heatmap(self, sample_df, tmp_path):
        """Test department heatmap creation."""
        report_gen = ReportGenerator(sample_df, str(tmp_path))
        fig = report_gen.create_department_heatmap()
        
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_create_ltv_cac_distribution(self, sample_df, tmp_path):
        """Test LTV/CAC distribution chart creation."""
        report_gen = ReportGenerator(sample_df, str(tmp_path))
        fig = report_gen.create_ltv_cac_distribution()
        
        assert fig is not None
    
    def test_save_figures(self, sample_df, tmp_path):
        """Test saving figures to disk."""
        report_gen = ReportGenerator(sample_df, str(tmp_path))
        report_gen.create_revenue_vs_cost_trend()
        
        saved_files = report_gen.save_all_figures(prefix='test')
        
        assert len(saved_files) >= 1
        assert all(Path(f).exists() for f in saved_files)
    
    def test_generate_json_metrics(self, sample_df, tmp_path):
        """Test JSON metrics generation."""
        report_gen = ReportGenerator(sample_df, str(tmp_path))
        
        summary_data = {
            'total_revenue': 100000,
            'total_cost': 60000
        }
        
        json_path = report_gen.generate_json_metrics(summary_data)
        
        assert Path(json_path).exists()
        
        # Verify author is in JSON
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        assert data['author'] == "OGHENEOCHUKO EMMANUEL OGIDIAGBA"
    
    def test_author_constant(self):
        """Test that author constant is defined correctly."""
        assert ReportGenerator.AUTHOR == "OGHENEOCHUKO EMMANUEL OGIDIAGBA"


class TestDataIntegrity:
    """Test data integrity and validation rules."""
    
    def test_revenue_greater_than_cost(self, sample_df):
        """Test that Revenue >= Cost for all records."""
        invalid_records = sample_df[sample_df['Revenue'] < sample_df['Cost']]
        assert len(invalid_records) == 0
    
    def test_margin_calculation(self, sample_df):
        """Test that Gross_Margin = Revenue - Cost."""
        expected_margin = sample_df['Revenue'] - sample_df['Cost']
        assert (sample_df['Gross_Margin'] == expected_margin).all()
    
    def test_margin_pct_calculation(self, sample_df):
        """Test that Margin_Pct is calculated correctly."""
        expected_pct = (sample_df['Gross_Margin'] / sample_df['Revenue']) * 100
        # Allow small floating point differences
        assert np.allclose(sample_df['Margin_Pct'], expected_pct, rtol=0.01)
    
    def test_ltv_cac_ratio_calculation(self, sample_df):
        """Test that LTV_CAC_Ratio = LTV / CAC."""
        expected_ratio = sample_df['LTV'] / sample_df['CAC']
        # Allow small floating point differences
        assert np.allclose(sample_df['LTV_CAC_Ratio'], expected_ratio, rtol=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
