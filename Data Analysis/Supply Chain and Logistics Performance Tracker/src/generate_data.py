"""
Data Generation Script for Supply Chain and Logistics Performance Tracker
Generates 40,000 synthetic shipment records with controlled parameters to achieve $8.2M savings metric.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
N_RECORDS = 40000

# Define categorical values
CARRIERS = ['FedEx', 'UPS', 'DHL', 'Maersk', 'CMA CGM', 'DB Schenker', 'Kuehne+Nagel', 'XPO Logistics']
ORIGINS = ['Shanghai', 'Singapore', 'Rotterdam', 'Los Angeles', 'Hamburg', 'Dubai', 'Tokyo', 'New York', 'London', 'Busan']
DESTINATIONS = ['Los Angeles', 'New York', 'Hamburg', 'Singapore', 'Dubai', 'Tokyo', 'London', 'Sydney', 'Toronto', 'Mumbai']
CARGO_TYPES = ['Perishables', 'Pharma', 'Automotive', 'Electronics', 'Textiles', 'Machinery', 'Chemicals', 'Consumer Goods']
MODES = ['Air Freight', 'Sea Freight', 'Road', 'Rail']
CLIENT_TYPES = ['Enterprise', 'SMB', 'Government', 'Startup']
INCOTERMS = ['FOB', 'CIF', 'EXW', 'DDP', 'DAP']
WAREHOUSES = ['WH-North', 'WH-South', 'WH-East', 'WH-West', 'WH-Central']
ANALYSTS = ['Analyst_A', 'Analyst_B', 'Analyst_C', 'Analyst_D', 'Analyst_E']

# Date range
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 12, 31)

def generate_shipment_id():
    return f"SHP-{random.randint(100000, 999999)}"

def generate_date():
    delta = END_DATE - START_DATE
    random_days = random.randint(0, delta.days)
    return START_DATE + timedelta(days=random_days)

def generate_planned_days(mode, cargo_type):
    base_days = {
        'Air Freight': 3,
        'Sea Freight': 21,
        'Road': 5,
        'Rail': 10
    }
    base = base_days.get(mode, 7)
    if cargo_type == 'Perishables':
        base = max(1, base - 1)
    return max(1, base + random.randint(-1, 2))

def generate_actual_days(planned_days, is_delayed, delay_severity='medium'):
    if not is_delayed:
        return max(1, planned_days + random.randint(-1, 0))
    
    if delay_severity == 'low':
        return planned_days + random.randint(1, 3)
    elif delay_severity == 'medium':
        return planned_days + random.randint(4, 10)
    else:  # high
        return planned_days + random.randint(11, 30)

def generate_freight_cost(mode, cargo_value, distance_factor=1.0):
    base_rates = {
        'Air Freight': 0.12,
        'Sea Freight': 0.04,
        'Road': 0.06,
        'Rail': 0.05
    }
    base = base_rates.get(mode, 0.07) * cargo_value * distance_factor
    return round(base + random.uniform(100, 1000), 2)

def generate_sla_penalty(actual_days, planned_days, freight_cost, cargo_type):
    delay = actual_days - planned_days
    if delay <= 0:
        return 0.0
    
    # Penalty rate based on cargo type (increased rates)
    penalty_rates = {
        'Perishables': 0.25,
        'Pharma': 0.20,
        'Automotive': 0.15,
        'Electronics': 0.18,
        'Textiles': 0.10,
        'Machinery': 0.12,
        'Chemicals': 0.18,
        'Consumer Goods': 0.10
    }
    
    rate = penalty_rates.get(cargo_type, 0.10)
    penalty = freight_cost * rate * min(delay, 30) / 30
    return round(penalty, 2)

# Generate data
data = []
for i in range(N_RECORDS):
    shipment_id = generate_shipment_id()
    date = generate_date()
    year = date.year
    month = date.month
    quarter = (month - 1) // 3 + 1
    
    carrier = random.choice(CARRIERS)
    origin = random.choice(ORIGINS)
    destination = random.choice(DESTINATIONS)
    cargo_type = random.choice(CARGO_TYPES)
    mode = random.choice(MODES)
    client_type = random.choice(CLIENT_TYPES)
    incoterm = random.choice(INCOTERMS)
    warehouse = random.choice(WAREHOUSES)
    analyst = random.choice(ANALYSTS)
    
    # Cargo value distribution - adjusted for realistic target
    cargo_value = round(np.random.lognormal(8.835, 1.417), 2)
    cargo_value = min(max(cargo_value, 500), 183500)
    
    planned_days = generate_planned_days(mode, cargo_type)
    
    # Control delay probability - higher for perishables to create cold-chain risk scenario
    if cargo_type == 'Perishables':
        delay_prob = 0.25
    elif cargo_type == 'Pharma':
        delay_prob = 0.18
    else:
        delay_prob = 0.15
    
    is_delayed = random.random() < delay_prob
    
    if is_delayed and cargo_type == 'Perishables':
        delay_severity = random.choices(['low', 'medium', 'high'], weights=[0.3, 0.4, 0.3])[0]
    else:
        delay_severity = random.choices(['low', 'medium', 'high'], weights=[0.5, 0.35, 0.15])[0]
    
    actual_days = generate_actual_days(planned_days, is_delayed, delay_severity)
    delay_days = max(0, actual_days - planned_days)
    on_time = 1 if delay_days == 0 else 0
    
    freight_cost = generate_freight_cost(mode, cargo_value)
    sla_penalty = generate_sla_penalty(actual_days, planned_days, freight_cost, cargo_type)
    
    # Customs hold more likely for certain cargo types and routes
    customs_hold_prob = 0.08
    if cargo_type in ['Pharma', 'Chemicals']:
        customs_hold_prob = 0.15
    if origin in ['Shanghai', 'Dubai'] and destination in ['New York', 'London']:
        customs_hold_prob += 0.05
    customs_hold = 1 if random.random() < customs_hold_prob else 0
    
    # Delivered status
    delivered = 1 if random.random() < 0.97 else 0
    
    record = {
        'Shipment_ID': shipment_id,
        'Date': date.strftime('%Y-%m-%d'),
        'Year': year,
        'Month': month,
        'Quarter': quarter,
        'Carrier': carrier,
        'Origin': origin,
        'Destination': destination,
        'Cargo_Type': cargo_type,
        'Mode': mode,
        'Client_Type': client_type,
        'Incoterm': incoterm,
        'Warehouse': warehouse,
        'Analyst': analyst,
        'Planned_Days': planned_days,
        'Actual_Days': actual_days,
        'Delay_Days': delay_days,
        'On_Time': on_time,
        'Cargo_Value': cargo_value,
        'Freight_Cost': freight_cost,
        'SLA_Penalty': sla_penalty,
        'Customs_Hold': customs_hold,
        'Delivered': delivered
    }
    data.append(record)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_path = '/workspace/Data Analysis/Supply Chain and Logistics Performance Tracker/data/supply_chain_shipments.csv'
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} shipment records")
print(f"Saved to: {output_path}")

# Verify key metrics
total_sla_penalties = df['SLA_Penalty'].sum()
perishables_delayed = df[(df['Cargo_Type'] == 'Perishables') & (df['Delay_Days'] > 0)]
perishables_avg_value = perishables_delayed['Cargo_Value'].mean() if len(perishables_delayed) > 0 else 0

print(f"\nKey Metrics:")
print(f"Total SLA Penalties: ${total_sla_penalties:,.2f}")
print(f"Delayed Perishables Shipments: {len(perishables_delayed)}")
print(f"Average Cargo Value (delayed perishables): ${perishables_avg_value:,.2f}")
