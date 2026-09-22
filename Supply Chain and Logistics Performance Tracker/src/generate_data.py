"""Synthetic shipment data generator. No target savings or ranking is imposed."""
import os,random
from datetime import datetime,timedelta
import numpy as np,pandas as pd
SEED=42; N_RECORDS=40000; START_DATE=datetime(2022,1,1); END_DATE=datetime(2024,12,31)
CARRIERS=['FedEx','UPS','DHL','Maersk','CMA CGM','DB Schenker','Kuehne+Nagel','XPO Logistics']
ORIGINS=['Shanghai','Singapore','Rotterdam','Los Angeles','Hamburg','Dubai','Tokyo','New York','London','Busan']
DESTINATIONS=['Los Angeles','New York','Hamburg','Singapore','Dubai','Tokyo','London','Sydney','Toronto','Mumbai']
CARGO_TYPES=['Perishables','Pharma','Automotive','Electronics','Textiles','Machinery','Chemicals','Consumer Goods']
MODES=['Air Freight','Sea Freight','Road','Rail']; CLIENT_TYPES=['Enterprise','SMB','Government','Startup']; INCOTERMS=['FOB','CIF','EXW','DDP','DAP']; WAREHOUSES=['WH-North','WH-South','WH-East','WH-West','WH-Central']; ANALYSTS=['Analyst_A','Analyst_B','Analyst_C','Analyst_D','Analyst_E']
def generate_dataset(n=N_RECORDS,seed=SEED):
 rng=np.random.default_rng(seed); random.seed(seed); rows=[]; span=(END_DATE-START_DATE).days
 plan={'Air Freight':3,'Sea Freight':21,'Road':5,'Rail':10}; delay_p={'Air Freight':.10,'Sea Freight':.18,'Road':.14,'Rail':.13}
 cargo_p={'Perishables':.08,'Pharma':.06,'Automotive':.02,'Electronics':.01,'Textiles':0,'Machinery':.01,'Chemicals':.04,'Consumer Goods':.01}
 cost={'Air Freight':.12,'Sea Freight':.04,'Road':.06,'Rail':.05}; penalty_rate={'Perishables':.25,'Pharma':.20,'Automotive':.15,'Electronics':.18,'Textiles':.10,'Machinery':.12,'Chemicals':.18,'Consumer Goods':.10}
 for i in range(n):
  date=START_DATE+timedelta(days=int(rng.integers(0,span+1))); carrier=random.choice(CARRIERS); origin=random.choice(ORIGINS); dest=random.choice(DESTINATIONS); cargo=random.choice(CARGO_TYPES); mode=random.choice(MODES)
  planned=max(1,plan[mode]+int(rng.integers(-1,3))-(1 if cargo=='Perishables' else 0)); p=min(.65,max(.01,delay_p[mode]+cargo_p[cargo]))
  if origin in ('Shanghai','Dubai') and dest in ('New York','London'): p=min(.65,p+.04)
  delayed=rng.random()<p; sev=rng.choice(['low','medium','high'],p=[.55,.32,.13] if cargo not in ('Perishables','Pharma') else [.40,.40,.20])
  extra=int(rng.integers(1,4) if sev=='low' else rng.integers(4,11) if sev=='medium' else rng.integers(11,31)) if delayed else 0
  actual=planned+extra; delay=max(0,actual-planned); value=float(np.clip(rng.lognormal(8.7,1.25),500,180000)); freight=max(100,value*cost[mode]*float(rng.uniform(.75,1.35))+float(rng.uniform(100,1000))); penalty=freight*penalty_rate[cargo]*min(delay,30)/30 if delay else 0
  customs=int(rng.random() < .07+(.07 if cargo in ('Pharma','Chemicals') else 0)+(.04 if origin in ('Shanghai','Dubai') and dest in ('New York','London') else 0)); delivered=int(rng.random()>=.03)
  rows.append({'Shipment_ID':f'SHP-{i+1:06d}','Date':date.strftime('%Y-%m-%d'),'Year':date.year,'Month':date.month,'Quarter':(date.month-1)//3+1,'Carrier':carrier,'Origin':origin,'Destination':dest,'Cargo_Type':cargo,'Mode':mode,'Client_Type':random.choice(CLIENT_TYPES),'Incoterm':random.choice(INCOTERMS),'Warehouse':random.choice(WAREHOUSES),'Analyst':random.choice(ANALYSTS),'Planned_Days':planned,'Actual_Days':actual,'Delay_Days':delay,'On_Time':int(delay==0),'Cargo_Value':round(value,2),'Freight_Cost':round(freight,2),'SLA_Penalty':round(penalty,2),'Customs_Hold':customs,'Delivered':delivered})
 return pd.DataFrame(rows)
if __name__=='__main__':
 df=generate_dataset(); out=os.path.join(os.path.dirname(__file__),'..','data','supply_chain_shipments.csv'); os.makedirs(os.path.dirname(out),exist_ok=True); df.to_csv(out,index=False); print(f'Generated {len(df):,} records')
