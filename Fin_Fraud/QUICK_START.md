# Quick Start Guide

## Installation
```bash
pip install pandas numpy scipy matplotlib jupyter
```

## Generate Data (3 Ways)

### 1. Command Line (Fastest)
```bash
python enhanced_data_generator.py
```
Creates 216,960 transactions with 2% fraud in `./data/`

### 2. Python Script
```python
from enhanced_data_generator import DarkPoolDataGenerator

gen = DarkPoolDataGenerator()
df, fraud = gen.generate_enhanced_dataset(num_transactions=100000, fraud_rate=0.02)
```

### 3. Interactive Notebook
```bash
jupyter notebook generate_enhanced.ipynb
```
Run all cells for visualizations + data

## For Teachers

### Generate Student Data (No Labels)
```python
gen = DarkPoolDataGenerator()
df, fraud = gen.generate_enhanced_dataset(100000, 0.02)
df.to_csv('student_data.csv', index=False)  # Excludes fraud labels
# Save fraud metadata separately for grading
```

### Validate Generated Data
```bash
python validate_enhanced_data.py
```

## Output Files
- `data/enhanced_raw_data.csv` - Transaction data
- `data/fraud_patterns_metadata.csv` - Fraud labels (keep from students)

## Fraud Types
- **Structuring**: Just under $10k, $50k, $100k
- **Wash Trading**: Same amounts, rapid timing
- **Layering**: Round $50k amounts
- **Market Manipulation**: $500k-800k coordinated
- **After Hours**: 2-4 AM with round amounts

## Parameters
- `num_transactions`: Dataset size (default: 216,960)
- `fraud_rate`: Fraud percentage (default: 0.02 = 2%)
- `random_seed`: For reproducibility (default: 42)