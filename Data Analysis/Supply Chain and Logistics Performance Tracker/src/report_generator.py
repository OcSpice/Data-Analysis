"""Decision-oriented reporting."""
import os,json
from datetime import datetime
class ReportGenerator:
 def __init__(self,df,results,output_dir='reports'): self.df=df; self.r=results; self.output_dir=output_dir; os.makedirs(output_dir,exist_ok=True)
 def metrics(self):
  return {'total_shipments':len(self.df),'on_time_rate_pct':round(self.df.On_Time.mean()*100,2),'delay_rate_pct':round((self.df.Delay_Days>0).mean()*100,2),'avg_delay_days':round(self.df.Delay_Days.mean(),2),'customs_hold_rate_pct':round(self.df.Customs_Hold.mean()*100,2),'freight_cost_usd':round(self.df.Freight_Cost.sum(),2),'cargo_value_usd':round(self.df.Cargo_Value.sum(),2),'observed_sla_penalties_usd':round(self.df.SLA_Penalty.sum(),2),'scenario_35pct_usd':round(self.r['savings_analysis']['total_scenario_exposure_base_35pct'],2)}
 def save_json(self):
  p=os.path.join(self.output_dir,'key_metrics.json'); open(p,'w').write(json.dumps(self.metrics(),indent=2)); return p
 def save_summary(self):
  m=self.metrics(); h=self.r['delay_hotspots']
  text=f"""SUPPLY CHAIN & LOGISTICS PERFORMANCE TRACKER
Generated: {datetime.now():%Y-%m-%d %H:%M}
EXECUTIVE OVERVIEW
- Shipments analyzed: {m['total_shipments']:,}
- On-time delivery: {m['on_time_rate_pct']:.1f}%
- Delay rate: {m['delay_rate_pct']:.1f}%
- Average delay: {m['avg_delay_days']:.2f} days
- Customs hold rate: {m['customs_hold_rate_pct']:.1f}%

OBSERVED FINANCIAL METRICS
- Freight cost: ${m['freight_cost_usd']:,.2f}
- Cargo value: ${m['cargo_value_usd']:,.2f}
- SLA penalties recorded: ${m['observed_sla_penalties_usd']:,.2f}

SCENARIO ANALYSIS
Cold-chain figures are modeled exposure scenarios, not observed losses or guaranteed savings.
- 35% exposure scenario: ${m['scenario_35pct_usd']:,.2f}

DELAY PATTERNS
- Highest average-delay carrier segment: {h['highest_delay_carrier']}
- Highest average-delay mode segment: {h['highest_delay_mode']}
- Highest average-delay cargo segment: {h['highest_delay_cargo_type']}

INTERPRETATION
Results are descriptive associations within synthetic operational data and do not establish causality.
Carrier scoring uses transparent weights and excludes carriers with fewer than 100 shipments.
"""
  p=os.path.join(self.output_dir,'executive_summary.txt'); open(p,'w').write(text); return p
 def save_carrier_report(self):
  lines=['CARRIER PERFORMANCE SCORECARD','Minimum comparative sample: 100 shipments','']
  for x in self.r['carrier_scorecard']: lines.append(f"{x['Carrier']}: shipments={x['Total_Shipments']}, on_time={x['On_Time_Rate']:.1%}, avg_delay={x['Avg_Delay_Days']:.2f}d, penalty_rate={x['Penalty_Rate']:.2%}, score={x.get('Performance_Score')}, eligible={x['Score_Eligible']}")
  p=os.path.join(self.output_dir,'carrier_scorecard.txt'); open(p,'w').write('\n'.join(lines)); return p
 def generate_all_reports(self): return {'metrics_json':self.save_json(),'executive_summary':self.save_summary(),'carrier_report':self.save_carrier_report()}
