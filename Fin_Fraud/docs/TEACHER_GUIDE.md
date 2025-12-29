# Teacher's Guide: Synthetic Data Anomaly Detection

## Assignment Overview

Students receive unlabeled financial transaction data and must identify fraudulent patterns using unsupervised learning techniques.

## Learning Objectives

1. Apply anomaly detection algorithms (Isolation Forest, GMM, LOF)
2. Engineer features from temporal and transactional data
3. Evaluate detection performance without labels
4. Understand Benford's Law in fraud detection
5. Interpret clustering results in financial context

## Dataset Generation

### Quick Setup
```python
from enhanced_data_generator import DarkPoolDataGenerator

gen = DarkPoolDataGenerator(random_seed=42)

# Generate student dataset (no labels)
student_df = gen.generate_student_dataset(
    num_transactions=50000,
    fraud_rate=0.05,  # 5% fraud
    difficulty='intermediate'  # beginner/intermediate/advanced
)
```

### Difficulty Levels

| Level | Fraud Rate | Pattern Visibility | Recommended For |
|-------|------------|-------------------|-----------------|
| Beginner | 10% | Obvious (round amounts, odd hours) | Intro courses |
| Intermediate | 5% | Mixed subtlety | Standard courses |
| Advanced | 2.5% | Subtle patterns | Advanced/graduate |

## Fraud Patterns (Answer Key)

### Pattern Types & Detection Hints

1. **Structuring (30%)**
   - Amounts: $9,999.99, $49,999.99, $99,999.99
   - Detection: Histogram peaks just below thresholds
   - Feature: `amount % 10000 == 9999.99`

2. **Wash Trading (20%)**
   - Identical amounts in 1-2 minute windows
   - Detection: Time-series clustering
   - Feature: Rolling window duplicate detection

3. **Layering (20%)**
   - Exact $50,000 amounts
   - Detection: Round number frequency analysis
   - Feature: `amount == 50000`

4. **Market Manipulation (20%)**
   - Large coordinated trades ($500k-$800k)
   - Detection: Outlier detection on amounts
   - Feature: Z-score > 3

5. **After Hours (10%)**
   - Trades at 2-4 AM with round amounts
   - Detection: Temporal anomaly detection
   - Feature: `hour in [2,3,4] AND amount % 1000 == 0`

## Evaluation Rubric

### Detection Performance (40%)

| Score | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| A (36-40) | >70% | >60% | >0.65 |
| B (32-35) | 60-70% | 50-60% | 0.55-0.65 |
| C (28-31) | 50-60% | 40-50% | 0.45-0.55 |
| D (24-27) | 40-50% | 30-40% | 0.35-0.45 |
| F (<24) | <40% | <30% | <0.35 |

### Feature Engineering (25%)

| Score | Criteria |
|-------|----------|
| Excellent (23-25) | Temporal features, merchant profiles, amount patterns, interaction features |
| Good (20-22) | Basic temporal + amount features |
| Satisfactory (17-19) | Simple statistical features only |
| Poor (<17) | Minimal or no feature engineering |

### Algorithm Selection (20%)

| Score | Criteria |
|-------|----------|
| Excellent (18-20) | Multiple algorithms with ensemble, proper tuning |
| Good (16-17) | 2+ algorithms with comparison |
| Satisfactory (14-15) | Single algorithm, well-implemented |
| Poor (<14) | Poor implementation or inappropriate choice |

### Analysis & Visualization (15%)

| Score | Criteria |
|-------|----------|
| Excellent (14-15) | Comprehensive visualizations, pattern interpretation |
| Good (12-13) | Clear plots, basic interpretation |
| Satisfactory (10-11) | Minimal visualization |
| Poor (<10) | No meaningful visualization |

## Grading Workflow

### 1. Generate Answer Key
```python
# Load student predictions
student_predictions = pd.read_csv('student_submission.csv')

# Load true labels (teacher only)
fraud_metadata = pd.read_csv('./data/fraud_patterns_metadata.csv')
true_labels = np.zeros(len(student_df))
true_labels[fraud_metadata['index']] = 1

# Calculate metrics
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, student_predictions, average='binary'
)
```

### 2. Pattern-Specific Evaluation
```python
# Check specific pattern detection
structuring_indices = fraud_metadata[
    fraud_metadata['type'] == 'structuring']['index']
structuring_recall = np.mean(
    student_predictions[structuring_indices] == 1
)

print(f"Structuring Detection Rate: {structuring_recall:.2%}")
```

### 3. Automated Grading Script
```python
def grade_submission(predictions, true_metadata, weights=None):
    """
    Automated grading with detailed feedback
    """
    weights = weights or {
        'detection': 0.40,
        'features': 0.25,
        'algorithms': 0.20,
        'analysis': 0.15
    }
    
    # Calculate detection score
    detection_score = calculate_detection_score(predictions, true_metadata)
    
    # Feature engineering (manual review needed)
    feature_score = 0  # Requires code review
    
    # Algorithm complexity (manual review)
    algorithm_score = 0  # Requires code review
    
    # Visualization quality (manual review)
    analysis_score = 0  # Requires code review
    
    total = (detection_score * weights['detection'] + 
             feature_score * weights['features'] +
             algorithm_score * weights['algorithms'] +
             analysis_score * weights['analysis'])
    
    return {
        'total': total,
        'detection': detection_score,
        'breakdown': {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }
```

## Common Student Mistakes

1. **Over-reliance on amount outliers**
   - Only catches market manipulation
   - Misses subtle patterns

2. **Ignoring temporal features**
   - Misses after-hours fraud
   - Can't detect wash trading

3. **Not checking Benford's Law**
   - Misses statistical anomalies
   - Poor baseline comparison

4. **Single algorithm approach**
   - Limited pattern detection
   - No robustness check

## Recommended Solution Approach

```python
# 1. Feature Engineering
df['hour'] = pd.to_datetime(df['Timestamp']).dt.hour
df['is_round_1k'] = (df['Amount'] % 1000 == 0)
df['is_threshold'] = df['Amount'].isin([9999.99, 49999.99, 99999.99])
df['amount_zscore'] = np.abs((df['Amount'] - df['Amount'].mean()) / df['Amount'].std())

# 2. Multiple Detection Methods
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

# Method 1: Isolation Forest
iso = IsolationForest(contamination=0.05)
iso_predictions = iso.fit_predict(features)

# Method 2: GMM
gmm = GaussianMixture(n_components=2)
gmm_scores = gmm.fit_predict(features)

# 3. Ensemble
final_predictions = (iso_predictions == -1) | (gmm_scores == 1)
```

## Additional Resources

- Benford's Law: First digit analysis
- Isolation Forest: Tree-based anomaly detection
- GMM: Probabilistic clustering
- Time-series anomaly detection techniques

## Support Materials

1. **Starter notebook**: Basic data loading and EDA
2. **Evaluation script**: Automated grading
3. **Solution notebook**: Complete reference implementation
4. **Office hours topics**: Feature engineering, algorithm selection