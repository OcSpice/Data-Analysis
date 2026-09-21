# Supply Chain & Logistics Performance Tracker

End-to-end supply-chain analytics using synthetic shipment records from 2022–2024.

## Questions
- Where are delays concentrated by carrier, mode, cargo, route and time?
- Which dimensions are associated with lower on-time performance?
- Which cold-chain segments show the highest delay exposure?
- What financial exposure is implied under explicit assumptions?

## Method
**Data → validation → descriptive analytics → risk segmentation → scenario modeling → reporting**

The generator does **not** target a predetermined savings amount.

### Carrier scorecard
40% on-time rate, 30% normalized inverse average delay, and 30% normalized inverse penalty rate. Carriers with at least 100 shipments receive a comparative score. Component metrics remain visible.

### Cold-chain scenarios
Perishables and Pharma are treated as cold-chain-sensitive. Delayed cargo value is exposed under 10%, 20%, 35%, and 50% assumptions. These are scenario estimates, not observed spoilage, realized losses, or guaranteed savings.

### Delay analysis
The project reports delay patterns and hotspots. It does not claim causal root causes.

## Limitations
Synthetic data; analyst-defined score weights; scenario assumptions require domain validation; descriptive associations do not establish causality.

## Run
```bash
python src/generate_data.py
python src/main.py
```
