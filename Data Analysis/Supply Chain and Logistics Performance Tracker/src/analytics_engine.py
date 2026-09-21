"""Supply-chain analytics with transparent scoring and scenario modeling."""
import numpy as np,pandas as pd
MIN_CARRIER_SHIPMENTS=100; COLD_CHAIN_TYPES=['Perishables','Pharma']; DEFAULT_EXPOSURE_SCENARIOS=(.10,.20,.35,.50)
class CarrierScorecard:
 def __init__(self,df): self.df=df.copy()
 def calculate_metrics(self):
  g=self.df.groupby('Carrier').agg(Total_Shipments=('Shipment_ID','count'),On_Time_Rate=('On_Time','mean'),Avg_Delay_Days=('Delay_Days','mean'),Total_SLA_Penalty=('SLA_Penalty','sum'),Total_Freight_Cost=('Freight_Cost','sum'),Total_Cargo_Value=('Cargo_Value','sum'),Total_Customs_Holds=('Customs_Hold','sum'),Delivery_Success_Rate=('Delivered','mean')).reset_index()
  g['Penalty_Rate']=np.where(g.Total_Freight_Cost>0,g.Total_SLA_Penalty/g.Total_Freight_Cost,0); g['Score_Eligible']=g.Total_Shipments>=MIN_CARRIER_SHIPMENTS; e=g[g.Score_Eligible].copy()
  if len(e):
   dm=max(e.Avg_Delay_Days.max(),1e-9); pm=max(e.Penalty_Rate.max(),1e-9); e['Performance_Score']=100*(.40*e.On_Time_Rate+.30*(1-e.Avg_Delay_Days/dm)+.30*(1-e.Penalty_Rate/pm)); e['Rank']=e.Performance_Score.rank(ascending=False,method='min').astype(int); g=g.merge(e[['Carrier','Performance_Score','Rank']],on='Carrier',how='left')
  else: g['Performance_Score']=np.nan; g['Rank']=np.nan
  return g.sort_values(['Score_Eligible','Performance_Score'],ascending=[False,False])
 def get_top_performers(self,n=3): return self.calculate_metrics().query('Score_Eligible').head(n)
class ColdChainRiskAnalyzer:
 def __init__(self,df): self.df=df.copy(); self.cold_chain_df=df[df.Cargo_Type.isin(COLD_CHAIN_TYPES)].copy()
 def identify_high_risk_routes(self):
  if self.cold_chain_df.empty:return pd.DataFrame()
  r=self.cold_chain_df.groupby(['Origin','Destination']).agg(Total_Shipments=('Shipment_ID','count'),Avg_Delay_Days=('Delay_Days','mean'),On_Time_Rate=('On_Time','mean'),Customs_Hold_Rate=('Customs_Hold','mean'),Total_Cargo_Value=('Cargo_Value','sum'),Total_SLA_Penalty=('SLA_Penalty','sum')).reset_index(); d=max(r.Avg_Delay_Days.max(),1e-9); r['Risk_Score']=40*(1-r.On_Time_Rate)+30*r.Customs_Hold_Rate+30*r.Avg_Delay_Days/d; r['Risk_Level']=pd.cut(r.Risk_Score,[-np.inf,30,60,np.inf],labels=['Low','Medium','High']); return r.sort_values('Risk_Score',ascending=False)
 def analyze_by_mode(self):
  if self.cold_chain_df.empty:return pd.DataFrame()
  return self.cold_chain_df.groupby('Mode').agg(Total_Shipments=('Shipment_ID','count'),Avg_Delay_Days=('Delay_Days','mean'),On_Time_Rate=('On_Time','mean'),Customs_Hold_Rate=('Customs_Hold','mean'),Total_SLA_Penalty=('SLA_Penalty','sum')).reset_index().sort_values('Avg_Delay_Days',ascending=False)
 def get_delay_exposure_proxy(self):
  d=self.cold_chain_df[self.cold_chain_df.Delay_Days>0].copy()
  if d.empty:return {'shipment_count':0,'total_cargo_value':0.0,'delay_exposure_index':0.0}
  d['Delay_Severity_Index']=(d.Delay_Days/d.Delay_Days.max()).clip(0,1); return {'shipment_count':len(d),'total_cargo_value':float(d.Cargo_Value.sum()),'delay_exposure_index':float((d.Cargo_Value*d.Delay_Severity_Index).sum())}
class ScenarioAnalyzer:
 def __init__(self,df): self.df=df.copy()
 def cold_chain_exposure(self,rates=DEFAULT_EXPOSURE_SCENARIOS):
  d=self.df[(self.df.Cargo_Type.isin(COLD_CHAIN_TYPES))&(self.df.Delay_Days>0)]; base=float(d.Cargo_Value.sum()); return pd.DataFrame([{'Assumed_Exposure_Rate':r,'Estimated_Exposure_USD':base*r,'Affected_Shipments':len(d),'Delayed_Cargo_Value_USD':base} for r in rates])
 def summary(self):
  s=self.cold_chain_exposure(); return {'scenario_table':s.to_dict('records'),'observed_sla_penalties':float(self.df.loc[self.df.Delay_Days>0,'SLA_Penalty'].sum()),'note':'Scenario estimates are assumptions, not observed losses or realized savings.'}
class DelayAnalyzer:
 def __init__(self,df): self.df=df.copy()
 def get_monthly_delay_trends(self):
  x=self.df.groupby(['Year','Month']).agg(Total_Shipments=('Shipment_ID','count'),Total_Delay_Days=('Delay_Days','sum'),Avg_Delay_Days=('Delay_Days','mean'),On_Time_Rate=('On_Time','mean')).reset_index(); x['Delay_Rate']=1-x.On_Time_Rate; return x.sort_values(['Year','Month'])
 def analyze_delays_by_category(self,category):
  x=self.df.groupby(category).agg(Total_Shipments=('Shipment_ID','count'),Total_Delay_Days=('Delay_Days','sum'),Avg_Delay_Days=('Delay_Days','mean'),On_Time_Rate=('On_Time','mean'),Total_SLA_Penalty=('SLA_Penalty','sum')).reset_index(); x['Delay_Rate']=1-x.On_Time_Rate; return x.sort_values('Avg_Delay_Days',ascending=False)
 def identify_delay_hotspots(self):
  return {'highest_delay_carrier':self.analyze_delays_by_category('Carrier').iloc[0].Carrier,'highest_delay_mode':self.analyze_delays_by_category('Mode').iloc[0].Mode,'highest_delay_cargo_type':self.analyze_delays_by_category('Cargo_Type').iloc[0].Cargo_Type,'overall_avg_delay':float(self.df.Delay_Days.mean()),'overall_delay_rate':float((self.df.Delay_Days>0).mean())}
class LogisticsAnalyticsEngine:
 def __init__(self,df): self.df=df.copy(); self.carrier_scorecard=CarrierScorecard(df); self.cold_chain_analyzer=ColdChainRiskAnalyzer(df); self.scenario_analyzer=ScenarioAnalyzer(df); self.delay_analyzer=DelayAnalyzer(df)
 def run_full_analysis(self):
  carrier=self.carrier_scorecard.calculate_metrics(); routes=self.cold_chain_analyzer.identify_high_risk_routes(); scenarios=self.scenario_analyzer.summary(); hotspots=self.delay_analyzer.identify_delay_hotspots()
  scenario_35=next(x['Estimated_Exposure_USD'] for x in scenarios['scenario_table'] if np.isclose(x['Assumed_Exposure_Rate'],0.35)); return {'carrier_scorecard':carrier.to_dict('records'),'top_carriers':self.carrier_scorecard.get_top_performers().to_dict('records'),'high_risk_routes':routes.to_dict('records'),'mode_analysis':self.cold_chain_analyzer.analyze_by_mode().to_dict('records'),'delay_exposure':self.cold_chain_analyzer.get_delay_exposure_proxy(),'scenario_analysis':scenarios,'savings_analysis':{'observed_sla_penalties':scenarios['observed_sla_penalties'],'total_scenario_exposure_base_35pct':float(scenario_35)},'delay_trends':self.delay_analyzer.get_monthly_delay_trends().to_dict('records'),'delay_hotspots':hotspots}
