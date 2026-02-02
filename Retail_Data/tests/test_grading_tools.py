"""Tests for grading_tools module."""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grading_tools import (
    evaluate_regression_performance,
    analyze_prediction_quality,
    generate_regression_report,
    create_prediction_template
)


class TestEvaluateRegressionPerformance:
    """Tests for regression performance evaluation."""

    def test_returns_dict(self, sample_retail_data, sample_predictions, temp_dir):
        """Test that function returns a dictionary."""
        # Save test files
        actual_file = os.path.join(temp_dir, 'actual.csv')
        pred_file = os.path.join(temp_dir, 'predictions.csv')

        sample_retail_data.to_csv(actual_file, index=False)
        sample_predictions.to_csv(pred_file, index=False)

        result = evaluate_regression_performance(pred_file, actual_file)
        assert isinstance(result, dict)

    def test_contains_required_metrics(self, sample_retail_data, sample_predictions, temp_dir):
        """Test that result contains all required metrics."""
        actual_file = os.path.join(temp_dir, 'actual.csv')
        pred_file = os.path.join(temp_dir, 'predictions.csv')

        sample_retail_data.to_csv(actual_file, index=False)
        sample_predictions.to_csv(pred_file, index=False)

        result = evaluate_regression_performance(pred_file, actual_file)

        required_keys = ['r2_score', 'mae', 'mse', 'rmse', 'total_predictions']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_perfect_predictions(self, sample_retail_data, temp_dir):
        """Test metrics with perfect predictions."""
        # Create perfect predictions
        perfect_predictions = pd.DataFrame({
            'Transaction_ID': sample_retail_data['Transaction_ID'],
            'predicted_amount': sample_retail_data['Total_Amount']
        })

        actual_file = os.path.join(temp_dir, 'actual.csv')
        pred_file = os.path.join(temp_dir, 'predictions.csv')

        sample_retail_data.to_csv(actual_file, index=False)
        perfect_predictions.to_csv(pred_file, index=False)

        result = evaluate_regression_performance(pred_file, actual_file)

        # R2 should be 1.0 (or very close)
        assert result['r2_score'] > 0.999
        # MAE and RMSE should be 0 (or very close)
        assert result['mae'] < 0.01
        assert result['rmse'] < 0.01

    def test_reasonable_predictions(self, sample_retail_data, sample_predictions, temp_dir):
        """Test metrics with reasonable predictions."""
        actual_file = os.path.join(temp_dir, 'actual.csv')
        pred_file = os.path.join(temp_dir, 'predictions.csv')

        sample_retail_data.to_csv(actual_file, index=False)
        sample_predictions.to_csv(pred_file, index=False)

        result = evaluate_regression_performance(pred_file, actual_file)

        # R2 should be positive (predictions correlated with actual)
        assert result['r2_score'] > 0
        # MAE should be reasonable
        assert result['mae'] > 0
        assert result['mae'] < sample_retail_data['Total_Amount'].mean()

    def test_handles_length_mismatch(self, sample_retail_data, temp_dir):
        """Test error handling for length mismatch."""
        # Create predictions with different length
        short_predictions = pd.DataFrame({
            'Transaction_ID': ['TXN-00001', 'TXN-00002'],
            'predicted_amount': [100, 200]
        })

        actual_file = os.path.join(temp_dir, 'actual.csv')
        pred_file = os.path.join(temp_dir, 'predictions.csv')

        sample_retail_data.to_csv(actual_file, index=False)
        short_predictions.to_csv(pred_file, index=False)

        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_regression_performance(pred_file, actual_file)

    def test_handles_missing_column(self, sample_retail_data, temp_dir):
        """Test error handling for missing prediction column."""
        # Create predictions without required column
        bad_predictions = pd.DataFrame({
            'Transaction_ID': sample_retail_data['Transaction_ID'],
            'wrong_column': [100] * len(sample_retail_data)
        })

        actual_file = os.path.join(temp_dir, 'actual.csv')
        pred_file = os.path.join(temp_dir, 'predictions.csv')

        sample_retail_data.to_csv(actual_file, index=False)
        bad_predictions.to_csv(pred_file, index=False)

        with pytest.raises(ValueError, match="predicted_amount"):
            evaluate_regression_performance(pred_file, actual_file)

    def test_accuracy_thresholds(self, sample_retail_data, sample_predictions, temp_dir):
        """Test that accuracy threshold percentages are calculated."""
        actual_file = os.path.join(temp_dir, 'actual.csv')
        pred_file = os.path.join(temp_dir, 'predictions.csv')

        sample_retail_data.to_csv(actual_file, index=False)
        sample_predictions.to_csv(pred_file, index=False)

        result = evaluate_regression_performance(pred_file, actual_file)

        assert 'within_10_pct' in result
        assert 'within_25_pct' in result
        assert 'within_50_pct' in result

        # Thresholds should be percentages (0-100)
        assert 0 <= result['within_10_pct'] <= 100
        assert 0 <= result['within_25_pct'] <= 100
        assert 0 <= result['within_50_pct'] <= 100

        # Wider thresholds should have higher percentages
        assert result['within_50_pct'] >= result['within_25_pct']
        assert result['within_25_pct'] >= result['within_10_pct']


class TestAnalyzePredictionQuality:
    """Tests for segment-wise prediction quality analysis."""

    def test_returns_dataframe(self, sample_retail_data, sample_predictions, temp_dir):
        """Test that function returns a DataFrame."""
        actual_file = os.path.join(temp_dir, 'actual.csv')
        pred_file = os.path.join(temp_dir, 'predictions.csv')

        sample_retail_data.to_csv(actual_file, index=False)
        sample_predictions.to_csv(pred_file, index=False)

        result = analyze_prediction_quality(pred_file, actual_file)
        assert isinstance(result, pd.DataFrame)

    def test_contains_segment_types(self, sample_retail_data, sample_predictions, temp_dir):
        """Test that result contains different segment types."""
        actual_file = os.path.join(temp_dir, 'actual.csv')
        pred_file = os.path.join(temp_dir, 'predictions.csv')

        sample_retail_data.to_csv(actual_file, index=False)
        sample_predictions.to_csv(pred_file, index=False)

        result = analyze_prediction_quality(pred_file, actual_file)

        # Should have segment analysis
        assert 'segment_type' in result.columns
        assert 'segment' in result.columns
        assert 'r2' in result.columns

    def test_amount_quartile_analysis(self, sample_retail_data, sample_predictions, temp_dir):
        """Test that amount quartile analysis is included."""
        actual_file = os.path.join(temp_dir, 'actual.csv')
        pred_file = os.path.join(temp_dir, 'predictions.csv')

        sample_retail_data.to_csv(actual_file, index=False)
        sample_predictions.to_csv(pred_file, index=False)

        result = analyze_prediction_quality(pred_file, actual_file)

        # Should have amount quartile segments
        quartile_rows = result[result['segment_type'] == 'Amount_Quartile']
        assert len(quartile_rows) > 0


class TestGenerateRegressionReport:
    """Tests for regression report generation."""

    def test_generates_text_report(self):
        """Test that text report is generated."""
        results = {
            'r2_score': 0.75,
            'mae': 50.0,
            'mse': 5000.0,
            'rmse': 70.71,
            'mape': 0.15,
            'mean_residual': 5.0,
            'std_residual': 45.0,
            'within_10_pct': 35.0,
            'within_25_pct': 65.0,
            'within_50_pct': 90.0,
            'total_predictions': 1000,
            'actual_mean': 250.0,
            'actual_std': 100.0,
            'predicted_mean': 255.0,
            'predicted_std': 95.0
        }

        report = generate_regression_report(results, output_format='text')

        assert isinstance(report, str)
        assert 'REGRESSION MODEL GRADING REPORT' in report
        assert 'R2 Score' in report
        assert 'RMSE' in report
        assert 'Grade' in report.upper() or 'GRADE' in report

    def test_generates_json_report(self):
        """Test that JSON report is generated."""
        results = {
            'r2_score': 0.75,
            'mae': 50.0,
            'rmse': 70.71,
            'total_predictions': 1000
        }

        report = generate_regression_report(results, output_format='json')

        import json
        parsed = json.loads(report)
        assert parsed['r2_score'] == 0.75

    def test_grade_calculation_excellent(self):
        """Test grade calculation for excellent performance."""
        results = {
            'r2_score': 0.90,
            'mae': 20.0,
            'mse': 1000.0,
            'rmse': 31.62,
            'mape': 0.05,
            'mean_residual': 0.0,
            'std_residual': 20.0,
            'within_10_pct': 70.0,
            'within_25_pct': 90.0,
            'within_50_pct': 99.0,
            'total_predictions': 1000,
            'actual_mean': 250.0,
            'actual_std': 100.0,
            'predicted_mean': 250.0,
            'predicted_std': 100.0
        }

        report = generate_regression_report(results)
        assert 'Grade: A' in report or 'Letter Grade: A' in report

    def test_grade_calculation_poor(self):
        """Test grade calculation for poor performance."""
        results = {
            'r2_score': 0.20,
            'mae': 150.0,
            'mse': 30000.0,
            'rmse': 173.21,
            'mape': 0.50,
            'mean_residual': 50.0,
            'std_residual': 100.0,
            'within_10_pct': 10.0,
            'within_25_pct': 25.0,
            'within_50_pct': 50.0,
            'total_predictions': 1000,
            'actual_mean': 250.0,
            'actual_std': 100.0,
            'predicted_mean': 300.0,
            'predicted_std': 150.0
        }

        report = generate_regression_report(results)
        assert 'Grade: F' in report or 'Letter Grade: F' in report


class TestCreatePredictionTemplate:
    """Tests for prediction template creation."""

    def test_creates_template_file(self, sample_retail_data, temp_dir):
        """Test that template file is created."""
        input_file = os.path.join(temp_dir, 'input.csv')
        output_file = os.path.join(temp_dir, 'template.csv')

        sample_retail_data.to_csv(input_file, index=False)
        create_prediction_template(input_file, output_file)

        assert os.path.exists(output_file)

    def test_template_has_correct_columns(self, sample_retail_data, temp_dir):
        """Test that template has correct columns."""
        input_file = os.path.join(temp_dir, 'input.csv')
        output_file = os.path.join(temp_dir, 'template.csv')

        sample_retail_data.to_csv(input_file, index=False)
        create_prediction_template(input_file, output_file)

        template = pd.read_csv(output_file)

        assert 'Transaction_ID' in template.columns
        assert 'predicted_amount' in template.columns

    def test_template_has_correct_length(self, sample_retail_data, temp_dir):
        """Test that template has same number of rows as input."""
        input_file = os.path.join(temp_dir, 'input.csv')
        output_file = os.path.join(temp_dir, 'template.csv')

        sample_retail_data.to_csv(input_file, index=False)
        create_prediction_template(input_file, output_file)

        template = pd.read_csv(output_file)
        assert len(template) == len(sample_retail_data)

    def test_template_transaction_ids_match(self, sample_retail_data, temp_dir):
        """Test that transaction IDs match input data."""
        input_file = os.path.join(temp_dir, 'input.csv')
        output_file = os.path.join(temp_dir, 'template.csv')

        sample_retail_data.to_csv(input_file, index=False)
        create_prediction_template(input_file, output_file)

        template = pd.read_csv(output_file)

        pd.testing.assert_series_equal(
            template['Transaction_ID'].reset_index(drop=True),
            sample_retail_data['Transaction_ID'].reset_index(drop=True)
        )


class TestIntegration:
    """Integration tests for grading tools."""

    def test_full_grading_pipeline(self, sample_retail_data, sample_predictions, temp_dir):
        """Test full grading pipeline."""
        actual_file = os.path.join(temp_dir, 'actual.csv')
        pred_file = os.path.join(temp_dir, 'predictions.csv')

        sample_retail_data.to_csv(actual_file, index=False)
        sample_predictions.to_csv(pred_file, index=False)

        # Evaluate performance
        metrics = evaluate_regression_performance(pred_file, actual_file)

        # Analyze segments
        segments = analyze_prediction_quality(pred_file, actual_file)

        # Generate report
        report = generate_regression_report(metrics)

        # All should complete without error
        assert metrics['r2_score'] is not None
        assert len(segments) > 0
        assert len(report) > 0
