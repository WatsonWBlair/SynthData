# Data Specification

## Output Files

| File | Purpose |
| `enhanced_raw_data.csv` | Transaction data without labels |
| `fraud_patterns_metadata.csv` | Answer key with fraud indices (keep private) |

## Data Schema

**enhanced_raw_data.csv**
- `Timestamp`: ISO 8601 datetime of transaction
- `TransactionID`: Unique identifier (UUID format)
- `AccountID`: Integer 1-15 representing institutional accounts
- `Amount`: USD transaction amount (float, $50-$1,000,000)
- `Merchant`: Market maker code A-J
- `TransactionType`: Purchase, Transfer, or Withdrawal
- `Location`: Trading center (New York, London, Tokyo, San Francisco, Los Angeles)

**fraud_patterns_metadata.csv**
- `index`: Row number in transaction data
- `type`: Fraud pattern type (structuring, wash_trading, layering, etc.)
- `pattern`: Specific pattern description

## Market Maker Profiles

### Merchants (A-J)
- **HFT** (A,E): High-frequency, $25-30K avg, market hours
- **Block** (B,F): Large blocks, $150-180K avg, extended hours  
- **Arbitrage** (C,I): Medium frequency, $75-85K avg
- **Institutional** (D,H): Low frequency, $200-250K avg
- **Others** (G,J): Aggregator/Specialist, $45-120K avg

## Fraud Patterns

| Type | Frequency | Signature |
|------|-----------|-----------|
| Structuring | 30% | Amounts just under $10K/$50K/$100K thresholds |
| Wash Trading | 20% | Identical amounts in rapid sequence (1-2 min) |
| Layering | 20% | Suspiciously round $50K amounts |
| Market Manipulation | 20% | Coordinated $500-800K transactions |
| After Hours | 10% | Round $100K at 2-4 AM |

## Statistical Properties

### Benford's Law Compliance
Transaction amounts follow a power-law distribution with parameters:
- **Alpha**: 1.7-2.2 (varies by market maker type)
- **Scale**: 0.8-1.8 (adjusts for transaction size)
- **Rounding**: 30% probability of institutional rounding ($1K, $5K, $10K)

### Temporal Distribution
- **Market Hours** (9-17): 60% of normal transactions
- **Extended Hours** (6-20): 30% of normal transactions
- **After Hours**: 10% of normal, higher fraud concentration

### Amount Distribution
- **Minimum**: $50
- **Maximum**: $1,000,000
- **Mean**: Varies by market maker ($25K-$250K)
- **Natural variation**: ±5% random noise

## Generation Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| num_transactions | 1,000-10M | 216,960 | Dataset size |
| fraud_rate | 0.01-0.10 | 0.02 | Fraud percentage |
| random_seed | Any integer | 42 | For reproducibility |

## Validation Metrics

- **Benford's Law**: Chi-squared test < 50
- **P-value**: > 0.05 for natural distribution
- **Fraud Detection**: F1-score 0.55-0.75 achievable
- **Pattern Detection**: 50-70% recall per fraud type