# Performance Evaluation Tools

Tools for measuring fraud detection performance on synthetic datasets.

## Automated Performance Metrics

```python
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd
import numpy as np

def evaluate_detection_performance(predictions_file, true_metadata_file):
    """Calculate objective performance metrics for fraud detection."""
    predictions = pd.read_csv(predictions_file)['is_fraud'].values
    metadata = pd.read_csv(true_metadata_file)
    
    # Create true labels
    true_labels = np.zeros(len(predictions))
    true_labels[metadata['index']] = 1
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    
    return {
        'precision': precision, 'recall': recall, 'f1_score': f1,
        'true_positives': tp, 'false_positives': fp,
        'true_negatives': tn, 'false_negatives': fn
    }
```

## Pattern-Specific Detection

```python
def analyze_pattern_detection(predictions_file, metadata_file):
    """Measure detection rate for each fraud pattern type."""
    predictions = pd.read_csv(predictions_file)['is_fraud'].values
    metadata = pd.read_csv(metadata_file)
    
    results = {}
    for pattern in metadata['type'].unique():
        indices = metadata[metadata['type'] == pattern]['index'].values
        detection_rate = predictions[indices].mean()
        results[pattern] = detection_rate
    
    return results
```

## Usage Example

```bash
# Generate dataset with answer key
python enhanced_data_generator.py

# Evaluate submission
from grading_tools import evaluate_detection_performance
results = evaluate_detection_performance(
    'student_submission.csv',
    'data/fraud_patterns_metadata.csv'
)
```

## Metrics Interpretation

- **Precision**: Percentage of predicted frauds that were correct
- **Recall**: Percentage of actual frauds that were detected
- **F1 Score**: Balance between precision and recall
- **Pattern Detection**: Success rate for each fraud type

## Grading Considerations

Objective metrics provide only part of the assessment. Also consider:
- Code quality and documentation
- Feature engineering creativity
- Algorithm implementation and tuning
- Visualization and interpretation
- Understanding of limitations