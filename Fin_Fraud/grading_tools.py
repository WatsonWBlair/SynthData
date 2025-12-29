#!/usr/bin/env python3
"""
Grading Tools for Financial Fraud Detection Assignment
======================================================

Tools for evaluating student submissions against ground truth fraud labels.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score
import glob
import os
from typing import Dict, List, Tuple, Optional
import json


def evaluate_detection_performance(predictions_file: str, true_metadata_file: str) -> Dict:
    """
    Calculate objective performance metrics for fraud detection.
    
    Args:
        predictions_file: Path to student submission CSV with 'is_fraud' column
        true_metadata_file: Path to ground truth fraud metadata CSV
        
    Returns:
        Dictionary containing performance metrics
    """
    # Load predictions
    predictions_df = pd.read_csv(predictions_file)
    if 'is_fraud' not in predictions_df.columns:
        raise ValueError(f"Predictions file must contain 'is_fraud' column")
    predictions = predictions_df['is_fraud'].values
    
    # Load ground truth
    metadata = pd.read_csv(true_metadata_file)
    
    # Create true labels array
    true_labels = np.zeros(len(predictions))
    fraud_indices = metadata['index'].values
    # Ensure indices are within bounds
    valid_indices = fraud_indices[fraud_indices < len(true_labels)]
    true_labels[valid_indices] = 1
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0
    )
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    
    # Additional metrics
    try:
        # For probability scores if available
        if 'fraud_score' in predictions_df.columns:
            scores = predictions_df['fraud_score'].values
            auc_roc = roc_auc_score(true_labels, scores)
            auc_pr = average_precision_score(true_labels, scores)
        else:
            auc_roc = None
            auc_pr = None
    except:
        auc_roc = None
        auc_pr = None
    
    return {
        'precision': float(precision),
        'recall': float(recall), 
        'f1_score': float(f1),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'accuracy': float((tp + tn) / len(predictions)),
        'auc_roc': float(auc_roc) if auc_roc else None,
        'auc_pr': float(auc_pr) if auc_pr else None,
        'total_predictions': len(predictions),
        'total_fraud': int(true_labels.sum()),
        'predicted_fraud': int(predictions.sum())
    }


def analyze_pattern_detection(predictions_file: str, metadata_file: str) -> pd.DataFrame:
    """
    Measure detection rate for each fraud pattern type.
    
    Args:
        predictions_file: Path to student submission CSV
        metadata_file: Path to ground truth fraud metadata CSV
        
    Returns:
        DataFrame with detection rates per fraud type
    """
    # Load data
    predictions_df = pd.read_csv(predictions_file)
    predictions = predictions_df['is_fraud'].values
    metadata = pd.read_csv(metadata_file)
    
    results = []
    
    # Analyze each fraud type
    for pattern in metadata['type'].unique():
        pattern_data = metadata[metadata['type'] == pattern]
        indices = pattern_data['index'].values
        
        # Filter valid indices
        valid_indices = indices[indices < len(predictions)]
        
        if len(valid_indices) > 0:
            detected = predictions[valid_indices].sum()
            total = len(valid_indices)
            detection_rate = detected / total if total > 0 else 0
            
            results.append({
                'pattern': pattern,
                'detected': int(detected),
                'total': int(total),
                'detection_rate': float(detection_rate),
                'missed': int(total - detected)
            })
    
    return pd.DataFrame(results).sort_values('detection_rate', ascending=False)


def evaluate_multiple_submissions(submission_dir: str, true_metadata_file: str, 
                                 output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Evaluate multiple student submissions at once.
    
    Args:
        submission_dir: Directory containing student submission CSV files
        true_metadata_file: Path to ground truth fraud metadata
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
        student_id = filename.replace('.csv', '')
        
        try:
            # Evaluate submission
            metrics = evaluate_detection_performance(submission_path, true_metadata_file)
            metrics['student_id'] = student_id
            metrics['filename'] = filename
            metrics['status'] = 'success'
            
            # Get pattern detection rates
            pattern_results = analyze_pattern_detection(submission_path, true_metadata_file)
            metrics['patterns_detected'] = len(pattern_results[pattern_results['detection_rate'] > 0])
            metrics['avg_pattern_detection'] = pattern_results['detection_rate'].mean()
            
            results.append(metrics)
            print(f"✓ Evaluated {student_id}: F1={metrics['f1_score']:.3f}")
            
        except Exception as e:
            # Handle failed evaluations
            results.append({
                'student_id': student_id,
                'filename': filename,
                'status': 'error',
                'error_message': str(e)
            })
            print(f"✗ Error evaluating {student_id}: {e}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Sort by F1 score
    if 'f1_score' in results_df.columns:
        results_df = results_df.sort_values('f1_score', ascending=False)
    
    # Save results if requested
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    # Print summary statistics
    if len(results_df[results_df['status'] == 'success']) > 0:
        success_df = results_df[results_df['status'] == 'success']
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Successful evaluations: {len(success_df)}/{len(results_df)}")
        print(f"Average F1 Score: {success_df['f1_score'].mean():.3f}")
        print(f"Best F1 Score: {success_df['f1_score'].max():.3f}")
        print(f"Worst F1 Score: {success_df['f1_score'].min():.3f}")
        print(f"Average Precision: {success_df['precision'].mean():.3f}")
        print(f"Average Recall: {success_df['recall'].mean():.3f}")
    
    return results_df


def evaluate_team_dataset(team_id: str, data_dir: str = './data/student_datasets') -> Dict:
    """
    Evaluate a specific team's predictions for multi-dataset generation.
    
    Args:
        team_id: Team identifier (e.g., '001', '002')
        data_dir: Directory containing team datasets and answer keys
        
    Returns:
        Dictionary with evaluation results
    """
    # Construct file paths
    submission_file = os.path.join(data_dir, f'team_{team_id}_predictions.csv')
    answer_key_file = os.path.join(data_dir, f'answer_key_{team_id}.csv')
    
    if not os.path.exists(submission_file):
        raise FileNotFoundError(f"Submission not found: {submission_file}")
    if not os.path.exists(answer_key_file):
        raise FileNotFoundError(f"Answer key not found: {answer_key_file}")
    
    # Evaluate performance
    metrics = evaluate_detection_performance(submission_file, answer_key_file)
    pattern_analysis = analyze_pattern_detection(submission_file, answer_key_file)
    
    # Combine results
    results = {
        'team_id': team_id,
        'performance_metrics': metrics,
        'pattern_detection': pattern_analysis.to_dict('records')
    }
    
    return results


def generate_grade_report(results: Dict, output_format: str = 'text') -> str:
    """
    Generate a formatted grading report from evaluation results.
    
    Args:
        results: Dictionary from evaluate_detection_performance
        output_format: 'text' or 'json'
        
    Returns:
        Formatted report string
    """
    if output_format == 'json':
        return json.dumps(results, indent=2)
    
    # Text format report
    report = []
    report.append("="*60)
    report.append("FRAUD DETECTION GRADING REPORT")
    report.append("="*60)
    
    # Performance metrics
    report.append("\n📊 PERFORMANCE METRICS")
    report.append("-"*30)
    report.append(f"Precision:       {results['precision']:.3f}")
    report.append(f"Recall:          {results['recall']:.3f}")
    report.append(f"F1 Score:        {results['f1_score']:.3f}")
    report.append(f"Accuracy:        {results['accuracy']:.3f}")
    
    if results.get('auc_roc'):
        report.append(f"AUC-ROC:         {results['auc_roc']:.3f}")
    if results.get('auc_pr'):
        report.append(f"AUC-PR:          {results['auc_pr']:.3f}")
    
    # Confusion matrix
    report.append("\n📈 CONFUSION MATRIX")
    report.append("-"*30)
    report.append(f"True Positives:  {results['true_positives']:>6}")
    report.append(f"False Positives: {results['false_positives']:>6}")
    report.append(f"True Negatives:  {results['true_negatives']:>6}")
    report.append(f"False Negatives: {results['false_negatives']:>6}")
    
    # Summary
    report.append("\n📋 SUMMARY")
    report.append("-"*30)
    report.append(f"Total Transactions: {results['total_predictions']:,}")
    report.append(f"Actual Fraud:       {results['total_fraud']:,} ({results['total_fraud']/results['total_predictions']*100:.2f}%)")
    report.append(f"Predicted Fraud:    {results['predicted_fraud']:,} ({results['predicted_fraud']/results['total_predictions']*100:.2f}%)")
    
    # Grade calculation (simple rubric)
    f1 = results['f1_score']
    if f1 >= 0.65:
        grade = 'A'
        points = 90 + (f1 - 0.65) * 100  # 90-100
    elif f1 >= 0.55:
        grade = 'B'
        points = 80 + (f1 - 0.55) * 100  # 80-89
    elif f1 >= 0.45:
        grade = 'C'
        points = 70 + (f1 - 0.45) * 100  # 70-79
    elif f1 >= 0.35:
        grade = 'D'
        points = 60 + (f1 - 0.35) * 100  # 60-69
    else:
        grade = 'F'
        points = max(0, f1 * 171.4)  # 0-59
    
    report.append("\n🎓 GRADE")
    report.append("-"*30)
    report.append(f"Letter Grade: {grade}")
    report.append(f"Points: {min(100, max(0, points)):.1f}/100")
    
    report.append("\n" + "="*60)
    
    return "\n".join(report)


def main():
    """Command-line interface for grading tools."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Grade fraud detection submissions')
    parser.add_argument('predictions', help='Path to predictions CSV file')
    parser.add_argument('truth', help='Path to ground truth metadata CSV')
    parser.add_argument('--patterns', action='store_true', 
                       help='Show pattern-specific detection rates')
    parser.add_argument('--report', action='store_true',
                       help='Generate full grading report')
    
    args = parser.parse_args()
    
    # Evaluate submission
    results = evaluate_detection_performance(args.predictions, args.truth)
    
    if args.report:
        print(generate_grade_report(results))
    else:
        print(f"F1 Score: {results['f1_score']:.3f}")
        print(f"Precision: {results['precision']:.3f}")
        print(f"Recall: {results['recall']:.3f}")
    
    if args.patterns:
        print("\nPattern Detection:")
        pattern_df = analyze_pattern_detection(args.predictions, args.truth)
        print(pattern_df.to_string())


if __name__ == "__main__":
    main()