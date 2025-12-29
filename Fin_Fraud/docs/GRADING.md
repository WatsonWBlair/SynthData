# Performance Evaluation Tools

Tools for measuring fraud detection performance on synthetic datasets.

## Using the Grading Tools

The `grading_tools.py` module provides all evaluation functions:

```python
from grading_tools import (
    evaluate_detection_performance,
    analyze_pattern_detection,
    evaluate_multiple_submissions,
    generate_grade_report
)

# Evaluate a single submission
results = evaluate_detection_performance(
    'student_predictions.csv',
    'data/fraud_patterns_metadata.csv'
)

# Generate grade report
report = generate_grade_report(results)
print(report)
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

## Command-Line Usage

```bash
# Grade a single submission
invoke grade --submission student_predictions.csv --report

# Grade all submissions in a directory
invoke grade-all --submission-dir data/submissions

# Or use directly
python grading_tools.py predictions.csv truth.csv --report --patterns
```

## Metrics Interpretation

- **Precision**: Percentage of predicted frauds that were correct
- **Recall**: Percentage of actual frauds that were detected
- **F1 Score**: Balance between precision and recall
- **Pattern Detection**: Success rate for each fraud type

## Multiple Student Datasets

Generate unique datasets to prevent copying:

```bash
# Generate 20 unique datasets for different teams
invoke generate-student-datasets --count 20

# Files created:
# - data/student_datasets/team_001_data.csv (for Team 1)
# - data/student_datasets/answer_key_001.csv (instructor only)
# - data/student_datasets/dataset_mapping.csv (master list)
```

Each dataset has the same parameters but different random patterns, ensuring fair comparison while preventing direct result sharing.

## Grading Considerations

Objective metrics provide only part of the assessment. Also consider:
- Code quality and documentation
- Feature engineering creativity
- Algorithm implementation and tuning
- Visualization and interpretation
- Understanding of limitations