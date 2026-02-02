#!/usr/bin/env python3
"""
Grading Tools for Retail Sales Prediction Assignment
====================================================

Tools for evaluating student regression model submissions.
Evaluates predictions of Total_Amount based on transaction features.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error
)
import glob
import os
from typing import Dict, List, Optional
import json


def evaluate_regression_performance(predictions_file: str, actual_file: str,
                                    prediction_col: str = 'predicted_amount',
                                    actual_col: str = 'Total_Amount') -> Dict:
    """
    Calculate objective performance metrics for regression prediction.

    Args:
        predictions_file: Path to student submission CSV with predicted values
        actual_file: Path to ground truth CSV with actual values
        prediction_col: Column name for predictions (default: 'predicted_amount')
        actual_col: Column name for actual values (default: 'Total_Amount')

    Returns:
        Dictionary containing performance metrics
    """
    # Load predictions
    predictions_df = pd.read_csv(predictions_file)
    if prediction_col not in predictions_df.columns:
        raise ValueError(f"Predictions file must contain '{prediction_col}' column")

    # Load actual values
    actual_df = pd.read_csv(actual_file)
    if actual_col not in actual_df.columns:
        raise ValueError(f"Actual file must contain '{actual_col}' column")

    # Ensure same length
    if len(predictions_df) != len(actual_df):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions_df)}, actual={len(actual_df)}"
        )

    predictions = predictions_df[prediction_col].values
    actual = actual_df[actual_col].values

    # Remove any NaN values
    mask = ~(np.isnan(predictions) | np.isnan(actual))
    predictions = predictions[mask]
    actual = actual[mask]

    if len(predictions) == 0:
        raise ValueError("No valid predictions after removing NaN values")

    # Calculate metrics
    r2 = r2_score(actual, predictions)
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)

    # MAPE with protection against division by zero
    non_zero_mask = actual != 0
    if non_zero_mask.sum() > 0:
        mape = mean_absolute_percentage_error(actual[non_zero_mask], predictions[non_zero_mask])
    else:
        mape = np.nan

    # Additional statistics
    residuals = predictions - actual
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    # Percentage within different error thresholds
    abs_pct_error = np.abs(residuals / np.where(actual != 0, actual, 1)) * 100
    within_10_pct = (abs_pct_error <= 10).mean() * 100
    within_25_pct = (abs_pct_error <= 25).mean() * 100
    within_50_pct = (abs_pct_error <= 50).mean() * 100

    return {
        'r2_score': float(r2),
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape) if not np.isnan(mape) else None,
        'mean_residual': float(mean_residual),
        'std_residual': float(std_residual),
        'within_10_pct': float(within_10_pct),
        'within_25_pct': float(within_25_pct),
        'within_50_pct': float(within_50_pct),
        'total_predictions': len(predictions),
        'actual_mean': float(actual.mean()),
        'actual_std': float(actual.std()),
        'predicted_mean': float(predictions.mean()),
        'predicted_std': float(predictions.std())
    }


def analyze_prediction_quality(predictions_file: str, actual_file: str,
                               prediction_col: str = 'predicted_amount',
                               actual_col: str = 'Total_Amount') -> pd.DataFrame:
    """
    Analyze prediction quality across different segments.

    Args:
        predictions_file: Path to student submission CSV
        actual_file: Path to ground truth CSV
        prediction_col: Column name for predictions
        actual_col: Column name for actual values

    Returns:
        DataFrame with quality metrics by segment
    """
    predictions_df = pd.read_csv(predictions_file)
    actual_df = pd.read_csv(actual_file)

    # Merge datasets
    if 'Transaction_ID' in predictions_df.columns and 'Transaction_ID' in actual_df.columns:
        merged = predictions_df.merge(actual_df, on='Transaction_ID', suffixes=('_pred', '_actual'))
    else:
        # Assume same order
        merged = pd.concat([predictions_df, actual_df], axis=1)

    # Ensure we have both columns
    if prediction_col not in merged.columns:
        raise ValueError(f"Missing column: {prediction_col}")
    if actual_col not in merged.columns:
        raise ValueError(f"Missing column: {actual_col}")

    results = []

    # Analyze by category if available
    if 'Category' in merged.columns:
        for category in merged['Category'].unique():
            cat_data = merged[merged['Category'] == category]
            if len(cat_data) > 10:
                preds = cat_data[prediction_col].values
                actual = cat_data[actual_col].values

                results.append({
                    'segment_type': 'Category',
                    'segment': category,
                    'count': len(cat_data),
                    'r2': r2_score(actual, preds),
                    'mae': mean_absolute_error(actual, preds),
                    'rmse': np.sqrt(mean_squared_error(actual, preds))
                })

    # Analyze by region if available
    if 'Store_Region' in merged.columns:
        for region in merged['Store_Region'].unique():
            reg_data = merged[merged['Store_Region'] == region]
            if len(reg_data) > 10:
                preds = reg_data[prediction_col].values
                actual = reg_data[actual_col].values

                results.append({
                    'segment_type': 'Region',
                    'segment': region,
                    'count': len(reg_data),
                    'r2': r2_score(actual, preds),
                    'mae': mean_absolute_error(actual, preds),
                    'rmse': np.sqrt(mean_squared_error(actual, preds))
                })

    # Analyze by amount quartiles
    merged['amount_quartile'] = pd.qcut(merged[actual_col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_data = merged[merged['amount_quartile'] == quartile]
        if len(q_data) > 10:
            preds = q_data[prediction_col].values
            actual = q_data[actual_col].values

            results.append({
                'segment_type': 'Amount_Quartile',
                'segment': quartile,
                'count': len(q_data),
                'r2': r2_score(actual, preds),
                'mae': mean_absolute_error(actual, preds),
                'rmse': np.sqrt(mean_squared_error(actual, preds))
            })

    return pd.DataFrame(results)


def evaluate_multiple_submissions(submission_dir: str, actual_file: str,
                                  prediction_col: str = 'predicted_amount',
                                  actual_col: str = 'Total_Amount',
                                  output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Evaluate multiple student submissions at once.

    Args:
        submission_dir: Directory containing student submission CSV files
        actual_file: Path to ground truth CSV
        prediction_col: Column name for predictions
        actual_col: Column name for actual values
        output_file: Optional path to save results CSV

    Returns:
        DataFrame with evaluation results for all submissions
    """
    results = []

    # Find all CSV submissions
    submission_files = glob.glob(os.path.join(submission_dir, "*.csv"))

    print(f"Found {len(submission_files)} submissions to evaluate")

    for submission_path in submission_files:
        filename = os.path.basename(submission_path)
        student_id = filename.replace('.csv', '').replace('_predictions', '')

        try:
            # Evaluate submission
            metrics = evaluate_regression_performance(
                submission_path, actual_file,
                prediction_col, actual_col
            )
            metrics['student_id'] = student_id
            metrics['filename'] = filename
            metrics['status'] = 'success'

            results.append(metrics)
            print(f"  Evaluated {student_id}: R2={metrics['r2_score']:.3f}, "
                  f"RMSE=${metrics['rmse']:.2f}")

        except Exception as e:
            results.append({
                'student_id': student_id,
                'filename': filename,
                'status': 'error',
                'error_message': str(e)
            })
            print(f"  Error evaluating {student_id}: {e}")

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Sort by R2 score
    if 'r2_score' in results_df.columns:
        results_df = results_df.sort_values('r2_score', ascending=False)

    # Save results if requested
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    # Print summary statistics
    success_df = results_df[results_df['status'] == 'success']
    if len(success_df) > 0:
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Successful evaluations: {len(success_df)}/{len(results_df)}")
        print(f"Average R2 Score: {success_df['r2_score'].mean():.3f}")
        print(f"Best R2 Score: {success_df['r2_score'].max():.3f}")
        print(f"Worst R2 Score: {success_df['r2_score'].min():.3f}")
        print(f"Average RMSE: ${success_df['rmse'].mean():.2f}")
        print(f"Average MAE: ${success_df['mae'].mean():.2f}")

    return results_df


def evaluate_team_dataset(team_id: str, data_dir: str = './data/student_datasets',
                          prediction_col: str = 'predicted_amount',
                          actual_col: str = 'Total_Amount') -> Dict:
    """
    Evaluate a specific team's predictions for multi-dataset generation.

    Args:
        team_id: Team identifier (e.g., '001', '002')
        data_dir: Directory containing team datasets
        prediction_col: Column name for predictions
        actual_col: Column name for actual values

    Returns:
        Dictionary with evaluation results
    """
    # Construct file paths
    submission_file = os.path.join(data_dir, f'team_{team_id}_predictions.csv')
    actual_file = os.path.join(data_dir, f'team_{team_id}_data.csv')

    if not os.path.exists(submission_file):
        raise FileNotFoundError(f"Submission not found: {submission_file}")
    if not os.path.exists(actual_file):
        raise FileNotFoundError(f"Data file not found: {actual_file}")

    # Evaluate performance
    metrics = evaluate_regression_performance(
        submission_file, actual_file,
        prediction_col, actual_col
    )

    # Segment analysis
    segment_analysis = analyze_prediction_quality(
        submission_file, actual_file,
        prediction_col, actual_col
    )

    return {
        'team_id': team_id,
        'performance_metrics': metrics,
        'segment_analysis': segment_analysis.to_dict('records')
    }


def generate_regression_report(results: Dict, output_format: str = 'text') -> str:
    """
    Generate a formatted grading report from regression results.

    Args:
        results: Dictionary from evaluate_regression_performance
        output_format: 'text' or 'json'

    Returns:
        Formatted report string
    """
    if output_format == 'json':
        return json.dumps(results, indent=2)

    # Text format report
    report = []
    report.append("="*60)
    report.append("REGRESSION MODEL GRADING REPORT")
    report.append("="*60)

    # Performance metrics
    report.append("\nPERFORMANCE METRICS")
    report.append("-"*30)
    report.append(f"R2 Score:        {results['r2_score']:.4f}")
    report.append(f"RMSE:            ${results['rmse']:.2f}")
    report.append(f"MAE:             ${results['mae']:.2f}")
    if results.get('mape'):
        report.append(f"MAPE:            {results['mape']*100:.2f}%")
    report.append(f"MSE:             ${results['mse']:.2f}")

    # Residual analysis
    report.append("\nRESIDUAL ANALYSIS")
    report.append("-"*30)
    report.append(f"Mean Residual:   ${results['mean_residual']:.2f}")
    report.append(f"Std Residual:    ${results['std_residual']:.2f}")

    # Accuracy thresholds
    report.append("\nACCURACY THRESHOLDS")
    report.append("-"*30)
    report.append(f"Within 10%:      {results['within_10_pct']:.1f}%")
    report.append(f"Within 25%:      {results['within_25_pct']:.1f}%")
    report.append(f"Within 50%:      {results['within_50_pct']:.1f}%")

    # Summary statistics
    report.append("\nSUMMARY")
    report.append("-"*30)
    report.append(f"Total Predictions: {results['total_predictions']:,}")
    report.append(f"Actual Mean:       ${results['actual_mean']:.2f}")
    report.append(f"Predicted Mean:    ${results['predicted_mean']:.2f}")

    # Grade calculation based on R2 score
    r2 = results['r2_score']
    if r2 >= 0.85:
        grade = 'A'
        points = 90 + (r2 - 0.85) * 66.67  # 90-100
    elif r2 >= 0.70:
        grade = 'B'
        points = 80 + (r2 - 0.70) * 66.67  # 80-89
    elif r2 >= 0.55:
        grade = 'C'
        points = 70 + (r2 - 0.55) * 66.67  # 70-79
    elif r2 >= 0.40:
        grade = 'D'
        points = 60 + (r2 - 0.40) * 66.67  # 60-69
    else:
        grade = 'F'
        points = max(0, r2 * 150)  # 0-59

    report.append("\nGRADE")
    report.append("-"*30)
    report.append(f"Letter Grade: {grade}")
    report.append(f"Points: {min(100, max(0, points)):.1f}/100")

    # Interpretation
    report.append("\nINTERPRETATION")
    report.append("-"*30)
    if r2 >= 0.85:
        report.append("Excellent model explaining most variance")
    elif r2 >= 0.70:
        report.append("Good model with reasonable predictive power")
    elif r2 >= 0.55:
        report.append("Moderate model - consider additional features")
    elif r2 >= 0.40:
        report.append("Weak model - significant improvement needed")
    else:
        report.append("Poor model - fundamentally review approach")

    report.append("\n" + "="*60)

    return "\n".join(report)


def create_prediction_template(data_file: str, output_file: str) -> None:
    """
    Create a prediction template file for students.

    Args:
        data_file: Path to the input data CSV
        output_file: Path to save the template CSV
    """
    df = pd.read_csv(data_file)

    # Create template with Transaction_ID and empty prediction column
    template = pd.DataFrame({
        'Transaction_ID': df['Transaction_ID'],
        'predicted_amount': np.nan
    })

    template.to_csv(output_file, index=False)
    print(f"Prediction template saved to: {output_file}")
    print(f"Contains {len(template):,} rows for prediction")


def main():
    """Command-line interface for grading tools."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Grade retail sales prediction submissions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Evaluate a single submission
  python grading_tools.py predictions.csv actual_data.csv

  # Generate full report
  python grading_tools.py predictions.csv actual_data.csv --report

  # Evaluate multiple submissions in a directory
  python grading_tools.py --batch submissions/ actual_data.csv

  # Create prediction template for students
  python grading_tools.py --template data.csv template.csv
        ''')

    parser.add_argument('predictions', nargs='?', help='Path to predictions CSV file')
    parser.add_argument('actual', nargs='?', help='Path to actual data CSV')
    parser.add_argument('--report', action='store_true',
                       help='Generate full grading report')
    parser.add_argument('--segments', action='store_true',
                       help='Show segment-specific analysis')
    parser.add_argument('--batch', type=str, metavar='DIR',
                       help='Evaluate all submissions in directory')
    parser.add_argument('--template', type=str, nargs=2, metavar=('INPUT', 'OUTPUT'),
                       help='Create prediction template from input data')
    parser.add_argument('--prediction-col', type=str, default='predicted_amount',
                       help='Column name for predictions (default: predicted_amount)')
    parser.add_argument('--actual-col', type=str, default='Total_Amount',
                       help='Column name for actual values (default: Total_Amount)')

    args = parser.parse_args()

    if args.template:
        create_prediction_template(args.template[0], args.template[1])
        return

    if args.batch:
        if not args.actual:
            parser.error('--batch requires actual data file')
        evaluate_multiple_submissions(
            args.batch, args.actual,
            args.prediction_col, args.actual_col,
            output_file='./data/grading_results.csv'
        )
        return

    if not args.predictions or not args.actual:
        parser.error('Both predictions and actual files are required')

    # Evaluate single submission
    results = evaluate_regression_performance(
        args.predictions, args.actual,
        args.prediction_col, args.actual_col
    )

    if args.report:
        print(generate_regression_report(results))
    else:
        print(f"R2 Score: {results['r2_score']:.4f}")
        print(f"RMSE: ${results['rmse']:.2f}")
        print(f"MAE: ${results['mae']:.2f}")
        if results.get('mape'):
            print(f"MAPE: {results['mape']*100:.2f}%")

    if args.segments:
        print("\nSegment Analysis:")
        segment_df = analyze_prediction_quality(
            args.predictions, args.actual,
            args.prediction_col, args.actual_col
        )
        print(segment_df.to_string())


if __name__ == "__main__":
    main()
