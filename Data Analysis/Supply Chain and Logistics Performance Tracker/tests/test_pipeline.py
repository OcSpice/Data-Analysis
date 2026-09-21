"""Regression tests for Project 2 analytical assumptions."""
import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))
from generate_data import generate_dataset
from data_quality import DataQualityValidator
from analytics_engine import LogisticsAnalyticsEngine
from report_generator import ReportGenerator
from visualization import VisualizationEngine

def test_generated_data_schema():
    df=generate_dataset(2000)
    clean,report=DataQualityValidator().validate_and_clean(df)
    assert len(clean)==2000
    assert clean.Delay_Days.ge(0).all()
    assert (clean.On_Time==(clean.Delay_Days==0)).all()

def test_no_targeted_savings_metric():
    df=generate_dataset(2000)
    result=LogisticsAnalyticsEngine(df).run_full_analysis()
    assert 'target_savings' not in str(result).lower()
    assert len(result['scenario_analysis']['scenario_table'])==4

def test_carrier_threshold_and_scenarios():
    df=generate_dataset(2000)
    result=LogisticsAnalyticsEngine(df).run_full_analysis()
    assert all(x['Total_Shipments']>=100 for x in result['top_carriers'])
    assert {round(x['Assumed_Exposure_Rate'],2) for x in result['scenario_analysis']['scenario_table']}=={.10,.20,.35,.50}
    assert 'total_savings' not in result['savings_analysis']
    assert 'cold_chain_loss_prevention' not in result['savings_analysis']
    assert 'total_scenario_exposure_base_35pct' in result['savings_analysis']

def test_reporting_and_visualization_interfaces(tmp_path):
    df=generate_dataset(500)
    result=LogisticsAnalyticsEngine(df).run_full_analysis()
    report=ReportGenerator(df,result,str(tmp_path/'reports'))
    assert 'scenario_35pct_usd' in report.metrics()
    viz=VisualizationEngine(df,str(tmp_path/'viz'))
    assert set(viz.create_all_visualizations())=={'monthly_delay_rate','carrier_on_time','cold_chain_on_time'}
