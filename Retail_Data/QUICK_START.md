# Retail Data Generator - Quick Start Guide

## Overview

The Retail Data Generator creates synthetic retail transaction data with realistic patterns for educational purposes. The generated data includes regional differentiation, seasonal patterns, demographic correlations, and customer behavior patterns suitable for regression modeling tasks.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn pytest
```

Or install from the project's requirements.txt:

```bash
pip install -r requirements.txt
```

## Basic Usage

### Generate Default Dataset

Generate a default dataset with 50,000 transactions:

```bash
cd Retail_Data
python retail_data_generator.py
```

Output: `./data/retail_data.csv`

### Generate Custom Dataset

Specify custom parameters:

```bash
# Generate 100,000 transactions with custom seed
python retail_data_generator.py --transactions 100000 --seed 123

# Specify date range
python retail_data_generator.py --transactions 50000 --start-date 2022-01-01 --end-date 2023-12-31

# Custom output location
python retail_data_generator.py --output ./my_data/retail_transactions.csv
```

### Generate Multiple Student Datasets

Generate unique datasets for multiple student teams:

```bash
# Generate 10 unique datasets with 25,000 transactions each
python retail_data_generator.py --student-datasets 10 --transactions 25000
```

Output location: `./data/student_datasets/team_XXX_data.csv`

## Validate Generated Data

After generating data, validate the statistical patterns:

```bash
# Validate default data file
python validate_retail_data.py

# Validate specific file
python validate_retail_data.py --input ./data/retail_data.csv

# Skip visualizations
python validate_retail_data.py --no-visualize

# Generate text report
python validate_retail_data.py --report ./data/validation_report.txt
```

## Grading Student Submissions

Evaluate regression model predictions:

```bash
# Evaluate single submission
python grading_tools.py predictions.csv actual_data.csv

# Generate full grading report
python grading_tools.py predictions.csv actual_data.csv --report

# Show segment-specific analysis
python grading_tools.py predictions.csv actual_data.csv --segments

# Evaluate multiple submissions in batch
python grading_tools.py --batch submissions_dir/ actual_data.csv

# Create prediction template for students
python grading_tools.py --template data.csv template.csv
```

## Python API Usage

### Generate Data Programmatically

```python
from retail_data_generator import RetailDataGenerator
from datetime import datetime

# Initialize generator with reproducible seed
generator = RetailDataGenerator(random_seed=42)

# Generate dataset
df = generator.generate_retail_dataset(
    num_transactions=10000,
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2023, 12, 31)
)

# Save to file
df.to_csv('./data/my_dataset.csv', index=False)
```

### Validate Data Programmatically

```python
from validate_retail_data import validate_retail_dataset

# Run full validation
df, results = validate_retail_dataset(
    data_path='./data/retail_data.csv',
    visualize=True
)

# Access specific validation results
print(f"Regional ANOVA p-value: {results['regional']['anova_p_value']}")
print(f"Online repeat rate: {results['customer_id']['online_repeat_rate']}")
```

### Grade Submissions Programmatically

```python
from grading_tools import (
    evaluate_regression_performance,
    generate_regression_report
)

# Evaluate submission
metrics = evaluate_regression_performance(
    predictions_file='student_predictions.csv',
    actual_file='actual_data.csv',
    prediction_col='predicted_amount',
    actual_col='Total_Amount'
)

# Generate report
report = generate_regression_report(metrics)
print(report)
```

## Running Tests

Execute the test suite:

```bash
# Run all tests
pytest Retail_Data/tests/ -v

# Run with coverage
pytest Retail_Data/tests/ --cov=Retail_Data --cov-report=term-missing

# Run specific test file
pytest Retail_Data/tests/test_data_generator.py -v
```

## Generated Data Patterns

The generator creates data with these built-in patterns:

### Regional Patterns
| Region | Income Multiplier | Online Preference |
|--------|------------------|-------------------|
| West | 1.25x (High) | 55% |
| North | 1.15x | 50% |
| East | 1.0x (Baseline) | 45% |
| South | 0.85x (Lower) | 35% |

### Seasonal Patterns
- **Q4 Electronics**: +35% spike
- **Black Friday**: 2-2.5x multiplier
- **Weekend**: Category-specific boosts (1.1-1.3x)
- **Pay Cycle**: Month start (+25%), mid-month (-10%)

### Customer ID Behavior
- **Online**: ~70% repeat customers
- **In-Store**: ~70% transient (new IDs)

## File Structure

```
Retail_Data/
├── retail_data_generator.py   # Main generator
├── validate_retail_data.py    # Validation & visualization
├── grading_tools.py           # Regression evaluation
├── QUICK_START.md             # This file
├── tests/
│   ├── conftest.py
│   ├── test_data_generator.py
│   ├── test_validate_data.py
│   └── test_grading_tools.py
├── docs/
│   └── DATA_SPECIFICATION.md
└── data/                      # Generated output
```

## Troubleshooting

### NumPy Version Conflict

If you see NumPy compatibility errors:

```bash
pip install numpy<2.0.0 pandas --upgrade
```

### Missing Dependencies

Install all dependencies:

```bash
pip install -r requirements.txt
```

### File Not Found Errors

Ensure you're running from the correct directory:

```bash
cd /path/to/SynthData/Retail_Data
python retail_data_generator.py
```
