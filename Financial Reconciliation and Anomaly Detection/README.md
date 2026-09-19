# Financial Reconciliation and Anomaly Detection

**Portfolio Category:** Data Analysis (enhanced with Machine Learning techniques)

**Author:** Oghenochuko Emmanuel Ogidiagba

---

## Project Overview

This repository contains a production-ready Python pipeline for financial reconciliation and anomaly detection. It demonstrates senior-level data analysis capabilities including data quality assurance, PII masking for compliance, and advanced discrepancy detection using statistical and machine learning methods.

The project processes 35,000+ synthetic financial transactions spanning 2022-2024, identifying anomalies, quantifying financial exposure, and generating actionable audit priorities.

---

## Portfolio Context

This project is part of a comprehensive multi-project portfolio organized into three categories:

1. **Data Analysis** (this project) - Demonstrates data quality assurance, PII handling, and discrepancy detection enhanced with ML techniques
2. **Business Analysis** - Focus on stakeholder insights and business intelligence
3. **Data Science** - Advanced predictive modeling and deep learning applications

### Key Competencies Demonstrated

- Data review and quality assurance at scale
- PII and sensitive data handling for compliance
- Discrepancy and anomaly detection
- Statistical analysis and machine learning integration
- Automated reporting and visualization
- Production-ready code architecture

---

## Features

### 1. Data Quality Engine
- Strict schema validation with type checking
- Missing value detection and handling strategies
- Duplicate record identification
- Format consistency validation
- Comprehensive quality scoring (0-1 scale)

### 2. PII Masking Module
- Partial name masking (initials only)
- Transaction ID anonymization
- Hash-based pseudonymization
- Complete redaction options
- Audit logging of all masking operations

### 3. Advanced Anomaly Detection
- **Z-Score Statistical Method**: Identifies outliers beyond standard deviation thresholds
- **Isolation Forest**: Unsupervised ML algorithm for anomaly isolation
- **Combined Scoring**: Ensemble approach merging both methods
- **Audit Priority Tiering**: HIGH/MEDIUM/LOW/NORMAL classification

### 4. Automated Reporting
- Summary statistics report
- Department risk analysis visualizations
- Source system reliability charts
- Temporal trend analysis
- Financial exposure breakdowns

---

## Directory Structure

```
Financial Reconciliation and Anomaly Detection/
├── pipeline.py                 # Main entry point
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data_loader.py         # Data loading utilities
│   ├── data_quality_engine.py # Validation and cleaning
│   ├── pii_masker.py          # PII anonymization
│   ├── anomaly_detector.py    # ML-based anomaly detection
│   └── report_generator.py    # Visualization and reports
├── tests/
│   └── test_pipeline.py       # Pytest unit tests
├── reports/                    # Generated outputs
├── data/                       # Processed data storage
├── Financial_Reconciliation_Data.csv  # Source dataset
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Navigate to project directory
cd "Financial Reconciliation and Anomaly Detection"

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Running the Full Pipeline

```bash
python pipeline.py
```

This executes the complete workflow:
1. Loads transaction data
2. Validates data quality
3. Applies PII masking
4. Detects anomalies using Z-score and Isolation Forest
5. Creates audit priority tiers
6. Generates reports and visualizations

### Using Individual Components

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import DataLoader
from data_quality_engine import DataQualityEngine
from pii_masker import PIIMasker
from anomaly_detector import AnomalyDetector
from report_generator import ReportGenerator

# Load data
loader = DataLoader("Financial_Reconciliation_Data.csv")
df = loader.load_data()

# Validate quality
dqe = DataQualityEngine(df)
quality_report = dqe.run_full_validation()
print(f"Quality Score: {quality_report['quality_score']:.2%}")

# Apply PII masking
masker = PIIMasker(dqe.get_cleaned_data())
df_masked = masker.mask_all_pii()

# Detect anomalies
detector = AnomalyDetector(df_masked)
detector.detect_zscore_anomalies(threshold=3.0)
detector.detect_isolation_forest_anomalies(contamination=0.05)
tiering = detector.create_audit_priority_tiering()

# Generate reports
reporter = ReportGenerator(df_masked, detector, "reports")
reporter.generate_all_reports()
```

---

## Output Reports

The pipeline generates the following reports in the `reports/` directory:

| File | Description |
|------|-------------|
| `01_summary_report.txt` | Text summary of analysis findings |
| `02_department_risk_analysis.png` | Department risk visualization |
| `03_source_system_reliability.png` | Source system bubble chart |
| `04_temporal_trends.png` | Monthly trend analysis |
| `05_financial_exposure.png` | Financial exposure dashboard |
| `top_exposure_transactions.csv` | Top 20 highest discrepancy transactions |

---

## Anomaly Detection Methods

### Z-Score Method
Identifies transactions where the discrepancy percentage exceeds a specified number of standard deviations from the mean (default threshold: 3.0).

### Isolation Forest
An unsupervised machine learning algorithm that isolates anomalies by randomly partitioning features. Configurable contamination rate (default: 5%).

### Audit Priority Tiers

| Tier | Criteria |
|------|----------|
| **HIGH** | Flagged by both methods OR discrepancy > 10% |
| **MEDIUM** | Flagged by one method OR discrepancy 5-10% |
| **LOW** | Discrepancy 2-5% |
| **NORMAL** | Below threshold discrepancies |

---

## Testing

Run the test suite using pytest:

```bash
pytest tests/test_pipeline.py -v
```

Tests cover:
- Data loading and validation
- Schema checking and missing value handling
- PII masking operations
- Anomaly detection algorithms
- End-to-end pipeline integration

---

## Dataset Description

The dataset contains 35,000 synthetic financial transactions with the following columns:

| Column | Description |
|--------|-------------|
| Transaction_ID | Unique transaction identifier |
| Date | Transaction date (2022-2024) |
| Year/Month/Quarter | Temporal categorization |
| Department | Business department |
| Category | Transaction category |
| Transaction_Type | Debit/Credit |
| Source_System | Originating system |
| Region | Geographic region |
| Currency | Transaction currency |
| Analyst | Processing analyst (PII) |
| Expected_Amount | Budgeted amount |
| Actual_Amount | Actual transaction amount |
| Discrepancy | Difference (Actual - Expected) |
| Discrepancy_Pct | Percentage discrepancy |
| Status | Reconciliation status |

---

## Compliance Notes

### PII Handling
- Analyst names are masked to initials (e.g., "John Smith" becomes "J***** S****")
- Transaction IDs are partially masked showing only last 4 digits
- All masking operations are logged for audit purposes

### Data Privacy
This project demonstrates compliance-ready practices for handling sensitive financial data, suitable for environments requiring GDPR, HIPAA, or SOX compliance.

---

## Technical Stack

- **Python 3.8+**
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Isolation Forest algorithm
- **matplotlib**: Visualization
- **pytest**: Unit testing

---

## Author Notes

This project showcases my experience in:
- Reviewing large, sensitive datasets for accuracy and compliance
- Handling personally identifiable and financial records
- Reconciling budget data and cross-checking documentation
- Implementing zero-compliance-failure data pipelines

The modular, object-oriented design ensures maintainability and extensibility for production environments.

---

## License

This project is provided as a portfolio demonstration. All code is original work.

---

## Contact

For questions about this project or my portfolio, please refer to my professional contact information.
