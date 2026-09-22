"""Dashboard-ready visualization layer for the supply-chain project."""
import os
import matplotlib.pyplot as plt
class VisualizationEngine:
 def __init__(self,df,output_dir='visualizations'):
  self.df=df.copy(); self.output_dir=output_dir; os.makedirs(output_dir,exist_ok=True)
 def _save(self,fig,name):
  p=os.path.join(self.output_dir,name); fig.tight_layout(); fig.savefig(p,dpi=150,bbox_inches='tight'); plt.close(fig); return p
 def executive_kpis(self):
  return {'shipments':len(self.df),'on_time_rate':self.df.On_Time.mean(),'avg_delay_days':self.df.Delay_Days.mean(),'customs_hold_rate':self.df.Customs_Hold.mean(),'freight_cost':self.df.Freight_Cost.sum(),'sla_penalties':self.df.SLA_Penalty.sum()}
 def create_delay_trend(self):
  x=self.df.groupby(['Year','Month']).agg(Delay_Rate=('On_Time',lambda s:1-s.mean()),Avg_Delay_Days=('Delay_Days','mean')).reset_index(); x['Period']=x.Year.astype(str)+'-'+x.Month.astype(str).str.zfill(2)
  fig,ax=plt.subplots(figsize=(12,5)); ax.plot(x.Period,x.Delay_Rate*100); ax.set(title='Monthly Delay Rate',xlabel='Month',ylabel='Delay rate (%)'); ax.tick_params(axis='x',rotation=60); return self._save(fig,'monthly_delay_rate.png')
 def create_carrier_performance(self):
  x=self.df.groupby('Carrier').agg(Shipments=('Shipment_ID','count'),On_Time_Rate=('On_Time','mean'),Avg_Delay_Days=('Delay_Days','mean')).reset_index().sort_values('On_Time_Rate')
  fig,ax=plt.subplots(figsize=(10,5)); ax.barh(x.Carrier,x.On_Time_Rate*100); ax.set(title='Carrier On-Time Performance',xlabel='On-time rate (%)',ylabel='Carrier'); return self._save(fig,'carrier_on_time.png')
 def create_route_risk_table(self):
  x=self.df.groupby(['Origin','Destination']).agg(Shipments=('Shipment_ID','count'),On_Time_Rate=('On_Time','mean'),Avg_Delay_Days=('Delay_Days','mean'),Customs_Hold_Rate=('Customs_Hold','mean'),Cargo_Value=('Cargo_Value','sum')).reset_index(); return x.sort_values(['Avg_Delay_Days','Cargo_Value'],ascending=False)
 def create_cold_chain_view(self):
  x=self.df[self.df.Cargo_Type.isin(['Perishables','Pharma'])].groupby('Cargo_Type').agg(Shipments=('Shipment_ID','count'),On_Time_Rate=('On_Time','mean'),Avg_Delay_Days=('Delay_Days','mean'),Cargo_Value=('Cargo_Value','sum')).reset_index()
  fig,ax=plt.subplots(figsize=(8,5)); ax.bar(x.Cargo_Type,x.On_Time_Rate*100); ax.set(title='Cold-Chain Cargo: On-Time Performance',xlabel='Cargo type',ylabel='On-time rate (%)'); return self._save(fig,'cold_chain_on_time.png')
 def create_all_visualizations(self):
  return {'monthly_delay_rate':self.create_delay_trend(),'carrier_on_time':self.create_carrier_performance(),'cold_chain_on_time':self.create_cold_chain_view()}
