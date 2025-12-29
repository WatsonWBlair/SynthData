# Synthetic Financial Data Generator

Educational toolkit for anomaly detection in dark pool trading data.

## Components

| File | Purpose |
|------|---------|
| `enhanced_data_generator.py` | Core generator with fraud injection |
| `generate_enhanced.ipynb` | Interactive generation with visualizations |
| `validate_enhanced_data.py` | Statistical validation and Benford's Law testing |

## Architecture

### 10 Market Makers (A-J)
- **HFT** (A,E): High-frequency, $25-30K avg, market hours
- **Block** (B,F): Large blocks, $150-180K avg, extended hours  
- **Arbitrage** (C,I): Medium frequency, $75-85K avg
- **Institutional** (D,H): Low frequency, $200-250K avg
- **Others** (G,J): Aggregator/Specialist, $45-120K avg

### 15 Account Profiles
Varying by: risk tolerance, size preference, activity level

### 5 Fraud Patterns

| Type | Frequency | Signature |
|------|-----------|-----------|
| Structuring | 30% | Amounts just under $10K/$50K/$100K thresholds |
| Wash Trading | 20% | Identical amounts in rapid sequence (1-2 min) |
| Layering | 20% | Suspiciously round $50K amounts |
| Market Manipulation | 20% | Coordinated $500-800K transactions |
| After Hours | 10% | Round $100K at 2-4 AM |

## Benford's Law Implementation

```python
# Power law distribution for natural-looking amounts
u = uniform(0.1, 1.0)
factor = (1 - u)^(-1/alpha)  # alpha varies by market maker
amount = base_amount * factor * scale

# 30% chance of institutional rounding
if random() < 0.3:
    round_to_nearest($1K, $5K, or $10K)
```

## Usage

### Basic Generation
```python
from enhanced_data_generator import DarkPoolDataGenerator

generator = DarkPoolDataGenerator(random_seed=42)
df, fraud_metadata = generator.generate_enhanced_dataset(
    num_transactions=100000,
    fraud_rate=0.02  # 2% fraud
)
```

### Student Dataset (No Labels)
```python
df, fraud = generator.generate_enhanced_dataset(100000, 0.02)
df.to_csv('student_data.csv', index=False)
# Keep fraud metadata separate for grading
pd.DataFrame(fraud).to_csv('answer_key.csv', index=False)
```

### Validation
```bash
python validate_enhanced_data.py
```

Expected output:
- Chi-squared < 50 (Benford compliance)
- P-value > 0.05 (natural distribution)

## Output Format

**enhanced_raw_data.csv**
- Timestamp, TransactionID, AccountID, Amount
- Merchant (A-J), TransactionType, Location

**fraud_patterns_metadata.csv** (Teacher only)
- index, type, pattern

## Educational Applications

1. **Unsupervised Learning**: Clustering, Isolation Forest, GMM
2. **Statistical Analysis**: Benford's Law, temporal patterns
3. **Feature Engineering**: Time-based features, merchant profiles
4. **Evaluation**: Precision/recall on hidden fraud labels

## Parameters

- `num_transactions`: Size (default: 216,960)
- `fraud_rate`: 0.01-0.10 (1-10%)
- `random_seed`: Reproducibility

## Notes

- Fraud patterns intentionally detectable for learning
- Real fraud detection requires more sophisticated methods
- Generated data approximates but doesn't replicate real markets