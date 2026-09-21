# Enterprise KPI Reporting Suite

**Author:** OGHENEOCHUKO EMMANUEL OGIDIAGBA  
**Portfolio Track:** Data Analysis

## Project Overview

An executive-style KPI reporting application demonstrating cross-functional
performance monitoring across Finance, Sales, HR, Operations, and regional
dimensions.

The project focuses on an auditable analytics workflow rather than a single
composite performance score:

**CSV → schema/data-quality checks → KPI aggregation → target variance →
management exceptions → visual reports → Streamlit dashboard**

The included dataset is a synthetic portfolio dataset. Reported revenue,
record counts, targets, and performance flags therefore describe this dataset
and its analytical assumptions; they are not claims about a real company.

## What the analysis does

### 1. KPI aggregation

The analytical engine calculates:

- Revenue, cost, gross margin, and revenue-to-cost ratio
- Weighted gross margin percentage
- Sales conversion derived from deals closed and leads generated
- LTV/CAC ratio
- Attrition, productivity, and training metrics
- SLA attainment, resolution time, and uptime
- Regional and quarterly performance

Where aggregation can change the meaning of a KPI, the engine avoids a simple
mean. For example, department margin is calculated as total gross margin
divided by total revenue, and conversion is derived from aggregate deals and
leads.

### 2. Target and variance analysis

The project uses department-specific **illustrative management thresholds**.
They are explicitly treated as portfolio assumptions rather than universal
industry benchmarks.

| Department | KPI | Direction | Illustrative threshold |
|---|---|---:|---:|
| Finance | Margin % | ≥ | 40% |
| Finance | Revenue / Cost | ≥ | 1.0 |
| Sales | LTV / CAC | ≥ | 3.0 |
| Sales | Conversion % | ≥ | 15% |
| Sales | Deals Closed | ≥ | 5 |
| HR | Attrition % | ≤ | 12% |
| HR | Productivity % | ≥ | 70% |
| HR | Training Hours | ≥ | 20 |
| Operations | SLA Met | ≥ | 1.0 |
| Operations | Uptime % | ≥ | 99.5% |
| Operations | Resolution Hours | ≤ | 12 |

For each KPI the application reports:

- Actual
- Target
- Direction
- Absolute gap
- Relative gap vs target
- Meets Target / Below Target / No Data

No heterogeneous KPI values are averaged into a single department
"health score".

### 3. Management exceptions

Below-target KPI rows are converted into a management exception table with
an explicit rule-based priority:

- **High:** absolute relative gap ≥ 10%
- **Medium:** absolute relative gap ≥ 5% and < 10%
- **Low:** absolute relative gap < 5%

This priority is an analytical rule for the portfolio project, not a claim
about business risk.

The project also reports areas with the highest **below-target exposure** using
the dataset's source Status field. This is exposure reporting, not a
composite performance ranking.

### 4. Period-over-period analysis

Quarterly reporting includes:

- Revenue and cost
- Gross margin
- Revenue-to-cost ratio
- Selected operating KPIs
- Revenue percentage change from the previous quarter
- Margin percentage-point change from the previous quarter

The application does not invent a trend when the source data cannot support
one.

### 5. Data quality framework

Data quality is separated into four dimensions:

1. **Completeness** — missing-cell coverage
2. **Uniqueness** — unique Record_ID coverage
3. **Validity** — domain/range and business-rule checks
4. **Consistency** — reconciliation checks such as:
   - Gross Margin = Revenue - Cost
   - Margin % agrees with Gross Margin / Revenue
   - LTV/CAC agrees with LTV / CAC

An equal-weight overall quality score is shown only as a transparent summary
of these four dimensions.

### 6. Privacy

Employee IDs can be hashed with SHA-256 through the anonymization module before
public sharing or downstream reporting.

## Dashboard

The Streamlit application contains:

1. **Executive Overview**
2. **Finance**
3. **Sales**
4. **People**
5. **Operations**
6. **Regional Performance**
7. **Target & Variance Analysis**
8. **Data Quality**

The target analysis view uses target-relative variance rather than min-max
normalization. This prevents a KPI from appearing favorable merely because it
is high relative to other departments when it is still below its business
threshold.

## Data model perspective

The current CSV is intentionally wide for portfolio convenience. A production
BI implementation could normalize it into:

- **Dim Date**
- **Dim Department**
- **Dim Region**
- **Dim Product**
- **Dim Employee**
- **Dim Cost Center**
- **Fact Performance**

This project demonstrates the analytical logic that could sit behind that
dimensional model.

## Repository structure

~~~
Enterprise KPI Reporting Suite/
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # schema + data quality dimensions
│   ├── anonymizer.py         # Employee ID hashing/masking
│   ├── kpi_engine.py         # KPI, target variance, trends, exceptions
│   └── visualization.py      # Plotly reports
├── tests/
│   └── test_pipeline.py      # analytical and data-quality tests
├── data/
│   └── KPI_Suite_Data.csv    # synthetic source dataset
├── reports/                  # generated HTML/JSON outputs
├── dashboard.py              # Streamlit application
├── main.py                   # pipeline runner
├── requirements.txt
└── README.md
~~~

## Run locally

### Install

~~~
cd "Enterprise KPI Reporting Suite"
pip install -r requirements.txt
~~~

### Run the pipeline

~~~
python main.py
~~~

The pipeline validates the dataset, calculates KPIs and exceptions, generates
visual reports, and writes reports/executive_metrics.json.

### Run the dashboard

~~~
streamlit run dashboard.py
~~~

### Run tests

~~~
pytest tests/test_pipeline.py -v
~~~

## Limitations and assumptions

- The source dataset is synthetic.
- Management thresholds are illustrative portfolio assumptions.
- The source Status field is treated as an observed dataset label; the
  project does not claim that it was independently derived from all 11 KPI
  thresholds.
- Wide-table data modeling is used for simplicity; a dimensional model would
  be preferable for a production BI environment.
- KPI averages are not automatically equivalent to business-weighted measures;
  the engine uses weighted/derived calculations where appropriate.
- No causal conclusions are drawn from the KPI relationships.
- Exception priority is a transparent analytical rule, not a risk-management
  framework.

## Author

**OGHENEOCHUKO EMMANUEL OGIDIAGBA**

Data Analysis portfolio project demonstrating Python, pandas, KPI design,
data-quality validation, target variance analysis, visualization, testing,
and Streamlit reporting.
