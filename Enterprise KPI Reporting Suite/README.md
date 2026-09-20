# Enterprise KPI Reporting Suite

**Author:** OGHENEOCHUKO EMMANUEL OGIDIAGBA  
**Portfolio Track:** Data Analysis  
**Data Scope:** ~38,000 records | $1.4 billion revenue scope

## Project Overview

This repository contains a production-ready Python application for enterprise-wide KPI tracking and executive reporting. It demonstrates senior-level data analysis capabilities including cross-functional performance tracking, executive-level data visualization, and robust data quality assurance.

The suite tracks Finance, Sales, HR, and Operations metrics side by side against strict business targets, providing the kind of interactive dashboard that CFOs actually open every Monday morning.

## Features

### 1. Modular Architecture
- **Object-oriented design** with clear separation of concerns
- **Data loading and validation** module with schema enforcement
- **KPI analytics engine** calculating 12 core targets across 4 departments
- **Visualization module** generating insight-driven charts
- **Interactive Streamlit dashboard** for executive consumption

### 2. Data Quality and Anonymization
- Comprehensive data quality scoring
- Schema validation (e.g., Revenue must be >= Cost)
- Missing value handling with multiple strategies
- Employee ID anonymization using SHA-256 hashing for privacy compliance

### 3. KPI Analytics Engine
Calculates department-specific health scores based on 12 core targets:

| Department | Metrics | Targets |
|------------|---------|---------|
| Finance | Margin_Pct, Revenue vs Cost | Margin >= 40%, Revenue >= Cost |
| Sales | LTV_CAC_Ratio, Conv_Rate_Pct, Deals_Closed | LTV/CAC >= 3.0, Conv Rate >= 15% |
| HR | Attrition_Pct, Productivity_Pct, Training_Hrs | Attrition <= 12%, Productivity >= 70% |
| Operations | SLA_Met, Uptime_Pct, Resolution_Hrs | Uptime >= 99.5%, Resolution <= 12 hrs |

### 4. Automated Reporting
- Revenue vs Cost trend charts
- Departmental target achievement heatmaps
- LTV to CAC ratio distributions
- Regional performance comparisons
- Quarterly trend analysis
- Executive summary JSON exports with author metadata

### 5. Persistent Metadata
All generated reports and JSON metrics include author attribution ("OGHENEOCHUKO EMMANUEL OGIDIAGBA") as class-level constants, ensuring metadata persists across pipeline re-runs.

## Directory Structure

```
Enterprise KPI Reporting Suite/
├── src/
│   ├── __init__.py           # Package initialization with author metadata
│   ├── data_loader.py        # CSV loading, schema validation, quality checks
│   ├── anonymizer.py         # PII anonymization for privacy compliance
│   ├── kpi_engine.py         # KPI calculation engine with 12 core targets
│   └── visualization.py      # Chart generation and report export
├── tests/
│   └── test_pipeline.py      # Comprehensive pytest unit tests
├── data/
│   └── KPI_Suite_Data.csv    # Source dataset (~38,000 records)
├── reports/                   # Generated visualizations and JSON metrics
├── dashboard.py              # Streamlit interactive dashboard
├── main.py                   # Main pipeline runner
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or navigate to the project directory:
   ```bash
   cd "Enterprise KPI Reporting Suite"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```bash
   python -c "from src.data_loader import DataLoader; print('Installation successful')"
   ```

## Usage

### Running the Full Pipeline

Execute the main pipeline to generate all reports:

```bash
python main.py
```

This will:
1. Load and validate the dataset
2. Run data quality checks
3. Calculate all KPIs and health scores
4. Generate visualizations in the `reports/` folder
5. Export executive metrics JSON

### Running the Interactive Dashboard

Launch the Streamlit dashboard for interactive exploration:

```bash
streamlit run dashboard.py
```

The dashboard provides:
- **Executive Overview**: Key metrics, trends, and department health scores
- **Department Analysis**: Drill-down into specific department performance
- **Regional Performance**: Geographic comparison and underperforming areas
- **Target Achievement**: Heatmaps and target reference
- **Data Quality**: Quality scores and anonymization demo

### Running Unit Tests

Execute the test suite:

```bash
pytest tests/test_pipeline.py -v
```

Test coverage includes:
- Data loading and validation
- Anonymization functions
- KPI calculations
- Report generation
- Data integrity checks

## Dataset Description

The dataset contains approximately 38,000 enterprise records representing a $1.4 billion revenue scope. Key columns include:

- **Identifiers**: Record_ID, Date, Year, Month, Quarter, Department, Region
- **Financial**: Revenue, Cost, Gross_Margin, Margin_Pct
- **Sales**: Deals_Closed, Leads_Generated, Conv_Rate_Pct, CAC, LTV, LTV_CAC_Ratio
- **HR**: Headcount, Attrition_Pct, Training_Hrs, Productivity_Pct
- **Operations**: SLA_Met, Ticket_Volume, Resolution_Hrs, Uptime_Pct
- **Status**: Above Target / Below Target flags

## Key Business Insights

The pipeline quantifies enterprise performance against 12 core targets across four departments:

1. **Finance Health**: Overall margin percentage and revenue-to-cost ratio
2. **Sales Efficiency**: LTV to CAC ratio indicating customer acquisition efficiency
3. **HR Stability**: Attrition rates and productivity metrics
4. **Operations Excellence**: SLA compliance and system uptime

Records flagged as "Below Target" are identified for executive remediation, with gap analysis quantifying revenue or efficiency shortfalls.

## Author Attribution

This project was developed by **OGHENEOCHUKO EMMANUEL OGIDIAGBA** as part of a comprehensive Data Analysis portfolio. The author metadata is embedded as class-level constants throughout the codebase and appears in all generated reports.

## License

This project is provided as a portfolio demonstration piece. All rights reserved.

## Contact

For questions about this project or portfolio inquiries, please contact the author.

---

*Enterprise KPI Reporting Suite - Data Analysis Portfolio Track*  
*Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA*
