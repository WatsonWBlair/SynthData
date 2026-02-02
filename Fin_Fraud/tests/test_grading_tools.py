"""Tests for grading_tools.py"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grading_tools import (
    evaluate_detection_performance,
    analyze_pattern_detection,
    evaluate_multiple_submissions,
    evaluate_team_dataset,
    generate_grade_report
)


@pytest.fixture
def ground_truth_metadata(temp_dir):
    """Create ground truth metadata file."""
    metadata = pd.DataFrame({
        'index': [10, 25, 50, 75, 90],
        'type': ['structuring', 'wash_trading', 'layering', 'market_manipulation', 'after_hours'],
        'pattern': ['threshold_avoidance', 'circular_amounts', 'round_amount', 'large_coordinated', 'suspicious_timing']
    })
    path = os.path.join(temp_dir, 'fraud_metadata.csv')
    metadata.to_csv(path, index=False)
    return path


@pytest.fixture
def perfect_predictions(temp_dir):
    """Create predictions with perfect detection."""
    predictions = np.zeros(100, dtype=int)
    predictions[[10, 25, 50, 75, 90]] = 1  # Exact match with ground truth

    df = pd.DataFrame({
        'TransactionID': [f'txn_{i:04d}' for i in range(100)],
        'is_fraud': predictions
    })
    path = os.path.join(temp_dir, 'perfect_predictions.csv')
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def no_detection_predictions(temp_dir):
    """Create predictions with no detection (all zeros)."""
    df = pd.DataFrame({
        'TransactionID': [f'txn_{i:04d}' for i in range(100)],
        'is_fraud': np.zeros(100, dtype=int)
    })
    path = os.path.join(temp_dir, 'no_detection.csv')
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def partial_detection_predictions(temp_dir):
    """Create predictions with partial detection."""
    predictions = np.zeros(100, dtype=int)
    # Detect some fraud (2 of 5 true positives) and have some false positives
    predictions[[10, 25, 30, 40]] = 1  # 10, 25 are true positives; 30, 40 are false positives

    df = pd.DataFrame({
        'TransactionID': [f'txn_{i:04d}' for i in range(100)],
        'is_fraud': predictions
    })
    path = os.path.join(temp_dir, 'partial_predictions.csv')
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def predictions_with_scores(temp_dir):
    """Create predictions with probability scores."""
    np.random.seed(42)
    predictions = np.zeros(100, dtype=int)
    predictions[[10, 25, 50, 75, 90]] = 1

    # Generate scores higher for actual fraud
    scores = np.random.uniform(0, 0.3, 100)
    scores[[10, 25, 50, 75, 90]] = np.random.uniform(0.7, 1.0, 5)

    df = pd.DataFrame({
        'TransactionID': [f'txn_{i:04d}' for i in range(100)],
        'is_fraud': predictions,
        'fraud_score': scores
    })
    path = os.path.join(temp_dir, 'predictions_with_scores.csv')
    df.to_csv(path, index=False)
    return path


class TestEvaluateDetectionPerformance:
    """Test suite for evaluate_detection_performance function."""

    def test_perfect_detection(self, perfect_predictions, ground_truth_metadata):
        """Test metrics with perfect detection."""
        results = evaluate_detection_performance(perfect_predictions, ground_truth_metadata)

        assert results['precision'] == 1.0
        assert results['recall'] == 1.0
        assert results['f1_score'] == 1.0
        assert results['true_positives'] == 5
        assert results['false_positives'] == 0
        assert results['true_negatives'] == 95
        assert results['false_negatives'] == 0
        assert results['total_fraud'] == 5
        assert results['predicted_fraud'] == 5

    def test_no_detection(self, no_detection_predictions, ground_truth_metadata):
        """Test metrics with no detection (all zeros)."""
        results = evaluate_detection_performance(no_detection_predictions, ground_truth_metadata)

        assert results['precision'] == 0.0
        assert results['recall'] == 0.0
        assert results['f1_score'] == 0.0
        assert results['true_positives'] == 0
        assert results['false_positives'] == 0
        assert results['true_negatives'] == 95
        assert results['false_negatives'] == 5

    def test_partial_detection(self, partial_detection_predictions, ground_truth_metadata):
        """Test metrics with partial detection."""
        results = evaluate_detection_performance(partial_detection_predictions, ground_truth_metadata)

        # 2 true positives (10, 25), 2 false positives (30, 40), 3 false negatives (50, 75, 90)
        assert results['true_positives'] == 2
        assert results['false_positives'] == 2
        assert results['false_negatives'] == 3
        assert results['precision'] == 0.5  # 2/(2+2)
        assert results['recall'] == 0.4  # 2/(2+3)
        assert results['accuracy'] == pytest.approx(0.95)  # (2+93)/100

    def test_with_probability_scores(self, predictions_with_scores, ground_truth_metadata):
        """Test metrics calculation with probability scores."""
        results = evaluate_detection_performance(predictions_with_scores, ground_truth_metadata)

        assert results['auc_roc'] is not None
        assert results['auc_pr'] is not None
        assert 0 <= results['auc_roc'] <= 1
        assert 0 <= results['auc_pr'] <= 1

    def test_missing_is_fraud_column(self, temp_dir, ground_truth_metadata):
        """Test error handling when is_fraud column is missing."""
        df = pd.DataFrame({
            'TransactionID': [f'txn_{i:04d}' for i in range(100)],
            'prediction': np.zeros(100)  # Wrong column name
        })
        path = os.path.join(temp_dir, 'bad_predictions.csv')
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="must contain 'is_fraud' column"):
            evaluate_detection_performance(path, ground_truth_metadata)

    def test_all_fraud_predicted(self, temp_dir, ground_truth_metadata):
        """Test when all transactions are predicted as fraud."""
        df = pd.DataFrame({
            'TransactionID': [f'txn_{i:04d}' for i in range(100)],
            'is_fraud': np.ones(100, dtype=int)
        })
        path = os.path.join(temp_dir, 'all_fraud.csv')
        df.to_csv(path, index=False)

        results = evaluate_detection_performance(path, ground_truth_metadata)

        assert results['recall'] == 1.0  # All fraud caught
        assert results['precision'] == 0.05  # 5/100
        assert results['false_positives'] == 95


class TestAnalyzePatternDetection:
    """Test suite for analyze_pattern_detection function."""

    def test_all_patterns_detected(self, perfect_predictions, ground_truth_metadata):
        """Test pattern detection rates with perfect detection."""
        results = analyze_pattern_detection(perfect_predictions, ground_truth_metadata)

        assert len(results) == 5
        assert all(results['detection_rate'] == 1.0)
        assert all(results['missed'] == 0)

    def test_no_patterns_detected(self, no_detection_predictions, ground_truth_metadata):
        """Test pattern detection rates with no detection."""
        results = analyze_pattern_detection(no_detection_predictions, ground_truth_metadata)

        assert len(results) == 5
        assert all(results['detection_rate'] == 0.0)
        assert all(results['detected'] == 0)

    def test_partial_pattern_detection(self, partial_detection_predictions, ground_truth_metadata):
        """Test pattern detection with partial detection."""
        results = analyze_pattern_detection(partial_detection_predictions, ground_truth_metadata)

        # Check specific patterns - structuring (10) and wash_trading (25) were detected
        structuring = results[results['pattern'] == 'structuring'].iloc[0]
        wash_trading = results[results['pattern'] == 'wash_trading'].iloc[0]
        layering = results[results['pattern'] == 'layering'].iloc[0]

        assert structuring['detection_rate'] == 1.0
        assert wash_trading['detection_rate'] == 1.0
        assert layering['detection_rate'] == 0.0

    def test_result_columns(self, perfect_predictions, ground_truth_metadata):
        """Test result DataFrame has correct columns."""
        results = analyze_pattern_detection(perfect_predictions, ground_truth_metadata)

        expected_columns = {'pattern', 'detected', 'total', 'detection_rate', 'missed'}
        assert set(results.columns) == expected_columns


class TestEvaluateMultipleSubmissions:
    """Test suite for evaluate_multiple_submissions function."""

    def test_multiple_submissions(self, temp_dir, ground_truth_metadata):
        """Test evaluation of multiple submissions."""
        # Create submission directory
        submission_dir = os.path.join(temp_dir, 'submissions')
        os.makedirs(submission_dir)

        # Create multiple submission files
        for student_id in ['student_001', 'student_002', 'student_003']:
            predictions = np.zeros(100, dtype=int)
            if student_id == 'student_001':
                predictions[[10, 25, 50, 75, 90]] = 1  # Perfect
            elif student_id == 'student_002':
                predictions[[10, 25]] = 1  # Partial
            # student_003 has no detections

            df = pd.DataFrame({
                'TransactionID': [f'txn_{i:04d}' for i in range(100)],
                'is_fraud': predictions
            })
            df.to_csv(os.path.join(submission_dir, f'{student_id}.csv'), index=False)

        # Evaluate all submissions
        results = evaluate_multiple_submissions(submission_dir, ground_truth_metadata)

        assert len(results) == 3
        assert all(results['status'] == 'success')

        # Check sorting by F1 score (best first)
        assert results.iloc[0]['f1_score'] == 1.0  # Perfect student

    def test_output_file_creation(self, temp_dir, ground_truth_metadata):
        """Test that results can be saved to file."""
        submission_dir = os.path.join(temp_dir, 'submissions')
        os.makedirs(submission_dir)

        # Create one submission
        df = pd.DataFrame({
            'TransactionID': [f'txn_{i:04d}' for i in range(100)],
            'is_fraud': np.zeros(100, dtype=int)
        })
        df.to_csv(os.path.join(submission_dir, 'test_student.csv'), index=False)

        output_file = os.path.join(temp_dir, 'results.csv')
        results = evaluate_multiple_submissions(submission_dir, ground_truth_metadata, output_file)

        assert os.path.exists(output_file)
        saved_results = pd.read_csv(output_file)
        assert len(saved_results) == 1

    def test_error_handling(self, temp_dir, ground_truth_metadata):
        """Test handling of invalid submission files."""
        submission_dir = os.path.join(temp_dir, 'submissions')
        os.makedirs(submission_dir)

        # Create invalid submission (missing is_fraud column)
        df = pd.DataFrame({
            'TransactionID': [f'txn_{i:04d}' for i in range(100)],
            'wrong_column': np.zeros(100, dtype=int)
        })
        df.to_csv(os.path.join(submission_dir, 'bad_student.csv'), index=False)

        results = evaluate_multiple_submissions(submission_dir, ground_truth_metadata)

        assert len(results) == 1
        assert results.iloc[0]['status'] == 'error'
        assert 'error_message' in results.columns


class TestEvaluateTeamDataset:
    """Test suite for evaluate_team_dataset function."""

    def test_successful_evaluation(self, temp_dir):
        """Test successful team evaluation."""
        # Setup team files
        data_dir = os.path.join(temp_dir, 'team_data')
        os.makedirs(data_dir)

        # Create prediction file
        predictions = np.zeros(100, dtype=int)
        predictions[[10, 25, 50]] = 1
        pred_df = pd.DataFrame({
            'TransactionID': [f'txn_{i:04d}' for i in range(100)],
            'is_fraud': predictions
        })
        pred_df.to_csv(os.path.join(data_dir, 'team_001_predictions.csv'), index=False)

        # Create answer key
        answer_key = pd.DataFrame({
            'index': [10, 25, 50, 75, 90],
            'type': ['structuring', 'wash_trading', 'layering', 'market_manipulation', 'after_hours']
        })
        answer_key.to_csv(os.path.join(data_dir, 'answer_key_001.csv'), index=False)

        results = evaluate_team_dataset('001', data_dir)

        assert results['team_id'] == '001'
        assert 'performance_metrics' in results
        assert 'pattern_detection' in results
        assert results['performance_metrics']['recall'] == 0.6  # 3 of 5 detected

    def test_missing_submission(self, temp_dir):
        """Test error when submission file is missing."""
        data_dir = os.path.join(temp_dir, 'team_data')
        os.makedirs(data_dir)

        with pytest.raises(FileNotFoundError, match="Submission not found"):
            evaluate_team_dataset('999', data_dir)

    def test_missing_answer_key(self, temp_dir):
        """Test error when answer key is missing."""
        data_dir = os.path.join(temp_dir, 'team_data')
        os.makedirs(data_dir)

        # Create only the prediction file
        df = pd.DataFrame({
            'TransactionID': ['txn_0001'],
            'is_fraud': [0]
        })
        df.to_csv(os.path.join(data_dir, 'team_001_predictions.csv'), index=False)

        with pytest.raises(FileNotFoundError, match="Answer key not found"):
            evaluate_team_dataset('001', data_dir)


class TestGenerateGradeReport:
    """Test suite for generate_grade_report function."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        return {
            'precision': 0.8,
            'recall': 0.6,
            'f1_score': 0.685,
            'accuracy': 0.95,
            'true_positives': 30,
            'false_positives': 7,
            'true_negatives': 950,
            'false_negatives': 20,
            'auc_roc': 0.85,
            'auc_pr': 0.75,
            'total_predictions': 1000,
            'total_fraud': 50,
            'predicted_fraud': 37
        }

    def test_text_format(self, sample_results):
        """Test text format report generation."""
        report = generate_grade_report(sample_results, output_format='text')

        assert 'FRAUD DETECTION GRADING REPORT' in report
        assert 'Precision:' in report
        assert 'Recall:' in report
        assert 'F1 Score:' in report
        assert 'CONFUSION MATRIX' in report
        assert 'GRADE' in report

    def test_json_format(self, sample_results):
        """Test JSON format report generation."""
        report = generate_grade_report(sample_results, output_format='json')

        # Should be valid JSON
        parsed = json.loads(report)
        assert parsed['precision'] == 0.8
        assert parsed['recall'] == 0.6

    def test_grade_a_threshold(self):
        """Test grade A assignment (F1 >= 0.65)."""
        results = {
            'precision': 0.7, 'recall': 0.7, 'f1_score': 0.70,
            'accuracy': 0.95, 'true_positives': 35, 'false_positives': 15,
            'true_negatives': 945, 'false_negatives': 15,
            'total_predictions': 1000, 'total_fraud': 50, 'predicted_fraud': 50
        }
        report = generate_grade_report(results)
        assert 'Letter Grade: A' in report

    def test_grade_b_threshold(self):
        """Test grade B assignment (0.55 <= F1 < 0.65)."""
        results = {
            'precision': 0.6, 'recall': 0.6, 'f1_score': 0.60,
            'accuracy': 0.93, 'true_positives': 30, 'false_positives': 20,
            'true_negatives': 930, 'false_negatives': 20,
            'total_predictions': 1000, 'total_fraud': 50, 'predicted_fraud': 50
        }
        report = generate_grade_report(results)
        assert 'Letter Grade: B' in report

    def test_grade_c_threshold(self):
        """Test grade C assignment (0.45 <= F1 < 0.55)."""
        results = {
            'precision': 0.5, 'recall': 0.5, 'f1_score': 0.50,
            'accuracy': 0.90, 'true_positives': 25, 'false_positives': 25,
            'true_negatives': 925, 'false_negatives': 25,
            'total_predictions': 1000, 'total_fraud': 50, 'predicted_fraud': 50
        }
        report = generate_grade_report(results)
        assert 'Letter Grade: C' in report

    def test_grade_d_threshold(self):
        """Test grade D assignment (0.35 <= F1 < 0.45)."""
        results = {
            'precision': 0.4, 'recall': 0.4, 'f1_score': 0.40,
            'accuracy': 0.88, 'true_positives': 20, 'false_positives': 30,
            'true_negatives': 920, 'false_negatives': 30,
            'total_predictions': 1000, 'total_fraud': 50, 'predicted_fraud': 50
        }
        report = generate_grade_report(results)
        assert 'Letter Grade: D' in report

    def test_grade_f_threshold(self):
        """Test grade F assignment (F1 < 0.35)."""
        results = {
            'precision': 0.2, 'recall': 0.2, 'f1_score': 0.20,
            'accuracy': 0.85, 'true_positives': 10, 'false_positives': 40,
            'true_negatives': 910, 'false_negatives': 40,
            'total_predictions': 1000, 'total_fraud': 50, 'predicted_fraud': 50
        }
        report = generate_grade_report(results)
        assert 'Letter Grade: F' in report

    def test_without_auc_metrics(self):
        """Test report generation without AUC metrics."""
        results = {
            'precision': 0.8, 'recall': 0.6, 'f1_score': 0.685,
            'accuracy': 0.95, 'true_positives': 30, 'false_positives': 7,
            'true_negatives': 950, 'false_negatives': 20,
            'auc_roc': None, 'auc_pr': None,
            'total_predictions': 1000, 'total_fraud': 50, 'predicted_fraud': 37
        }
        report = generate_grade_report(results)

        assert 'AUC-ROC' not in report or 'None' not in report
        assert 'Precision:' in report  # Should still have basic metrics
