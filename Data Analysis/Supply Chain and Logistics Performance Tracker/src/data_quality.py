"""Data validation and anonymization."""
import hashlib,pandas as pd
class DataQualityValidator:
 REQUIRED_COLUMNS=['Shipment_ID','Date','Year','Month','Quarter','Carrier','Origin','Destination','Cargo_Type','Mode','Client_Type','Incoterm','Warehouse','Analyst','Planned_Days','Actual_Days','Delay_Days','On_Time','Cargo_Value','Freight_Cost','SLA_Penalty','Customs_Hold','Delivered']
 def __init__(self,tolerance_days=1): self.tolerance_days=tolerance_days
 def validate_and_clean(self,df):
  missing=[c for c in self.REQUIRED_COLUMNS if c not in df.columns]
  if missing: raise ValueError(f'Schema validation failed. Missing columns: {missing}')
  out=df.copy(); report={'original_rows':len(out),'validation_passed':True,'issues':{'missing_values':out.isna().sum().loc[lambda x:x>0].to_dict()}}; out['Date']=pd.to_datetime(out['Date'],errors='coerce'); bad_dates=int(out['Date'].isna().sum());
  if bad_dates: report['validation_passed']=False; errors=[f'{bad_dates} invalid Date values']
  else: errors=[]
  for c in ['Planned_Days','Actual_Days','Delay_Days','Cargo_Value','Freight_Cost','SLA_Penalty']:
   if out[c].isna().any(): out[c]=out[c].fillna(out[c].median())
  for c in ['Carrier','Cargo_Type','Mode','Client_Type','Incoterm','Warehouse','Analyst']:
   if out[c].isna().any():
    mode=out[c].mode()
    out[c]=out[c].fillna(mode.iat[0] if not mode.empty else 'Unknown')
  for c in ['On_Time','Customs_Hold','Delivered']:
   if out[c].isna().any(): out[c]=out[c].fillna(0)
  invalid=out.Actual_Days<out.Planned_Days-self.tolerance_days
  if invalid.any(): out.loc[invalid,'Actual_Days']=out.loc[invalid,'Planned_Days']; errors.append(f'{int(invalid.sum())} transit values corrected')
  expected=(out.Actual_Days-out.Planned_Days).clip(lower=0)
  if (out.Delay_Days!=expected).any(): out['Delay_Days']=expected; errors.append('Delay_Days recalculated')
  expected_ot=(out.Delay_Days==0).astype(int)
  if (out.On_Time!=expected_ot).any(): out['On_Time']=expected_ot; errors.append('On_Time recalculated')
  for c in ['Cargo_Value','Freight_Cost']:
   bad=out[c]<=0
   if bad.any(): report['validation_passed']=False; errors.append(f'{int(bad.sum())} non-positive {c} values')
  dup=int(out.duplicated().sum())
  if dup: out=out.drop_duplicates().reset_index(drop=True); errors.append(f'{dup} duplicate rows removed')
  report['issues']['integrity']=errors; report['duplicate_rows_removed']=dup; report['cleaned_rows']=len(out); return out,report
class DataAnonymizer:
 def __init__(self,salt='supply_chain_portfolio'): self.salt=salt
 def hash_value(self,value):
  if pd.isna(value): return 'ANONYMIZED'
  return hashlib.sha256(f'{self.salt}_{value}'.encode()).hexdigest()[:12]
 def create_anonymized_dataset(self,df):
  out=df.copy()
  if 'Analyst' in out:
   out['Analyst']=out.Analyst.map({x:f'Analyst_{i+1}' for i,x in enumerate(sorted(out.Analyst.dropna().unique()))})
  if 'Shipment_ID' in out: out['Shipment_ID']=out.Shipment_ID.map(self.hash_value)
  if 'Client_Type' in out:
   known={'Enterprise','SMB','Government','Startup'}
   out['Client_Type']=out['Client_Type'].where(out['Client_Type'].isin(known),'Other')
  return out
def load_and_validate_data(file_path): return DataQualityValidator().validate_and_clean(pd.read_csv(file_path))
