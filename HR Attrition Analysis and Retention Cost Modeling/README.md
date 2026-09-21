# HR Attrition Analysis and Retention Cost Modeling

An end-to-end **HR Data Analysis** project that combines workforce attrition analysis, leakage-safe predictive modeling, model explainability, and scenario-based replacement-cost analysis.

## Business Objective

The project examines employee attrition patterns and translates the findings into:

- workforce segmentation and descriptive insights
- a reproducible attrition prediction workflow
- model explanations using genuine SHAP values
- estimated replacement-cost exposure
- illustrative retention-effectiveness scenarios

The project deliberately distinguishes **observed associations**, **model predictions**, and **financial scenarios**. It does not claim that an observed feature causes attrition or that a retention program will generate a guaranteed saving.

## Dataset

- 1,470 employee records
- IBM HR Employee Attrition benchmark dataset
- Target: `Attrition`
- Publicly shared employee identifiers are masked before analysis outputs are generated.

## Analytical Workflow

### 1. Data quality

- schema validation
- missing-value checks
- duplicate checks
- categorical-value validation
- numerical-range validation
- employee identifier masking

### 2. Feature engineering

Derived features include:

- `AnnualIncome` = MonthlyIncome × 12
- `ReplacementCost` = AnnualIncome × configured replacement-cost multiplier
- `TenureBucket`
- `IsOverTime`
- `PromotionStagnation`
- `LowSatisfactionCount`
- `PoorWorkLifeBalance`

### 3. Workforce analysis

The descriptive analysis covers:

- overall attrition
- overtime vs non-overtime attrition
- department
- job role
- tenure
- selected workforce and financial segments

The overtime comparison is reported as an **observed attrition-rate ratio**, not a causal risk estimate.

### 4. Predictive modeling

Two models are evaluated:

1. **Logistic Regression** — interpretable baseline
2. **Random Forest** — nonlinear comparison model

Preprocessing is implemented with scikit-learn `ColumnTransformer` and `Pipeline`.

This prevents preprocessing statistics from being learned from the test set.

Evaluation metrics include:

- Accuracy
- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- Confusion matrix

Because attrition is imbalanced, ROC-AUC, PR-AUC, recall and F1 are reported alongside accuracy rather than relying on accuracy alone.

### 5. Explainability

The Random Forest is explained with **TreeSHAP** on held-out test observations.

The project reports:

- global mean absolute SHAP importance
- feature importance aggregated back to the original HR variables
- an example local prediction explanation

SHAP explanations describe how features contribute to the model's predictions. They are not causal explanations of employee behavior.

### 6. Business impact

Replacement-cost exposure is calculated as:

[
Replacement Cost = Annual Income 	imes Replacement Cost Multiplier
]

The default multiplier is **1.5× annual salary**.

The project reports the estimated replacement-cost exposure associated with observed departures.

It then performs illustrative retention scenarios:

- 10% effectiveness
- 20% effectiveness
- 30% effectiveness
- 40% effectiveness
- 50% effectiveness

For each scenario:

[
Avoided Cost Estimate =
Observed Replacement Cost Exposure
	imes
Scenario Effectiveness
]

These are **hypothetical sensitivity scenarios**, not observed or guaranteed savings.

## Key Business Questions

The analysis is designed to answer questions such as:

- What is the observed attrition rate?
- How does observed attrition differ by overtime status?
- Which departments and job roles have higher observed attrition?
- How does attrition vary across tenure bands?
- Which variables contribute most strongly to the Random Forest's predictions?
- What replacement-cost exposure is associated with observed departures?
- How would different hypothetical retention-effectiveness levels translate into avoided-cost estimates?

## Repository Structure

```text
HR Attrition Analysis and Retention Cost Modeling/
├── HR-Employee-Attrition-Dataset.csv
├── README.md
├── pipeline.py
├── requirements.txt
├── src/
│   ├── attrition_model.py
│   ├── business_impact.py
│   ├── data_loader.py
│   ├── data_validator.py
│   ├── exploratory_analysis.py
│   ├── feature_engineer.py
│   ├── pii_masker.py
│   └── report_generator.py
├── reports/
└── tests/
```

## Running the Project

From this project directory:

```bash
pip install -r requirements.txt
python pipeline.py
```

Generated reports are written to `reports/`.

## Outputs

The pipeline generates:

- `executive_summary.txt`
- `retention_plan.txt`
- `data_dictionary.txt`
- `key_metrics.json`
- segment-level attrition CSV files
- `model_comparison.csv`
- dynamic analysis visualizations

## Limitations

- The IBM HR dataset is a benchmark dataset and is not evidence about any specific employer's workforce.
- The analysis is observational; associations do not establish causation.
- Replacement-cost estimates depend on the selected salary multiplier.
- Retention scenarios are sensitivity estimates rather than measured intervention outcomes.
- Predictive probabilities are not guarantees that an employee will leave.
- Any operational HR use would require additional validation, monitoring, fairness assessment and organizational context.
- Model predictions should not be used as the sole basis for employment decisions.

## Technology

- Python
- pandas
- NumPy
- scikit-learn
- SHAP
- Matplotlib
- pytest

## Project Classification

**Primary category:** Data Analysis  
**Predictive component:** Supervised Machine Learning  
**Explainability:** TreeSHAP  
**Business layer:** Replacement-cost exposure and scenario analysis
