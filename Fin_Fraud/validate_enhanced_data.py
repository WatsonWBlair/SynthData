#!/usr/bin/env python3
"""
Enhanced Data Validation Script with Visualizations
Validates statistical compliance and Benford's Law effectiveness
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def validate_enhanced_dataset(visualize=True):
    """
    Comprehensive validation of generated dataset
    
    Args:
        visualize: Whether to create visualization plots
    
    Returns:
        Tuple of (dataframe, chi_squared, p_value, validation_results)
    """
    # Load enhanced dataset
    df = pd.read_csv('./data/enhanced_raw_data.csv')
    print(f'Enhanced dataset loaded: {len(df):,} transactions')
    
    # Initialize results dictionary
    validation_results = {}
    
    # 1. Benford's Law Analysis
    amounts = df['Amount'].values
    first_digits = [int(str(amount)[0]) for amount in amounts if str(amount)[0].isdigit()]
    
    # Expected Benford's distribution
    expected_benford = [np.log10(1 + 1/d) for d in range(1, 10)]
    
    # Observed distribution
    digit_counts = {d: first_digits.count(d) for d in range(1, 10)}
    total_count = len(first_digits)
    observed_props = [digit_counts[d]/total_count for d in range(1, 10)]
    
    # Chi-squared test
    expected_counts = [p * total_count for p in expected_benford]
    observed_counts = [digit_counts[d] for d in range(1, 10)]
    
    chi2, p_value = stats.chisquare(observed_counts, expected_counts)
    
    validation_results['benford'] = {
        'chi_squared': chi2,
        'p_value': p_value,
        'compliant': p_value > 0.05,
        'expected': expected_benford,
        'observed': observed_props
    }
    
    print('\n=== Benford\'s Law Compliance ===')
    print('Expected proportions:', [f'{p:.3f}' for p in expected_benford])
    print('Observed proportions:', [f'{p:.3f}' for p in observed_props])
    print(f'Chi-squared: {chi2:.2f}')
    print(f'P-value: {p_value:.4f}')
    
    if p_value > 0.05:
        print('✅ Dataset follows Benford\'s Law (natural distribution)')
    else:
        print('⚠️  Dataset deviates from Benford\'s Law (investigate further)')
    
    # 2. Dataset Statistics
    print(f'\n=== Dataset Summary ===')
    print(f'Total transactions: {len(df):,}')
    print(f'Amount range: ${df["Amount"].min():.2f} - ${df["Amount"].max():.2f}')
    print(f'Average amount: ${df["Amount"].mean():.2f}')
    print(f'Median amount: ${df["Amount"].median():.2f}')
    print(f'Std deviation: ${df["Amount"].std():.2f}')
    print(f'Unique merchants: {df["Merchant"].nunique()}')
    print(f'Unique accounts: {df["AccountID"].nunique()}')
    print(f'Unique locations: {df["Location"].nunique()}')
    
    validation_results['statistics'] = {
        'total_transactions': len(df),
        'amount_min': df["Amount"].min(),
        'amount_max': df["Amount"].max(),
        'amount_mean': df["Amount"].mean(),
        'amount_median': df["Amount"].median(),
        'amount_std': df["Amount"].std()
    }
    
    # 3. Temporal Pattern Validation
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    hourly_distribution = df['Hour'].value_counts().sort_index()
    
    # Check for realistic trading hours distribution
    market_hours = hourly_distribution[9:17].sum()
    total_hours = len(df)
    market_concentration = market_hours / total_hours
    
    print(f'\n=== Temporal Patterns ===')
    print(f'Market hours concentration: {market_concentration:.2%}')
    print(f'Peak hour: {hourly_distribution.idxmax()}:00 ({hourly_distribution.max()} transactions)')
    print(f'Quiet hour: {hourly_distribution.idxmin()}:00 ({hourly_distribution.min()} transactions)')
    
    validation_results['temporal'] = {
        'market_concentration': market_concentration,
        'peak_hour': int(hourly_distribution.idxmax()),
        'quiet_hour': int(hourly_distribution.idxmin())
    }
    
    # 4. Fraud Pattern Analysis
    try:
        fraud_df = pd.read_csv('./data/fraud_patterns_metadata.csv')
        print(f'\n=== Fraud Patterns ===')
        print(f'Total fraud patterns injected: {len(fraud_df):,}')
        print(f'Actual fraud rate: {len(fraud_df)/len(df)*100:.2f}%')
        
        fraud_types = fraud_df['type'].value_counts()
        print('\nFraud type distribution:')
        for fraud_type, count in fraud_types.items():
            print(f'  {fraud_type}: {count} ({count/len(fraud_df)*100:.1f}%)')
        
        validation_results['fraud'] = {
            'total_patterns': len(fraud_df),
            'fraud_rate': len(fraud_df)/len(df),
            'type_distribution': fraud_types.to_dict()
        }
        
    except FileNotFoundError:
        print('\n⚠️  No fraud metadata file found')
        fraud_df = None
    
    # 5. Merchant Profile Validation
    merchant_stats = df.groupby('Merchant')['Amount'].agg(['mean', 'std', 'count'])
    
    print(f'\n=== Merchant Profiles ===')
    print('Average amounts by merchant:')
    for merchant in sorted(merchant_stats.index):
        avg = merchant_stats.loc[merchant, 'mean']
        count = merchant_stats.loc[merchant, 'count']
        print(f'  Merchant {merchant}: ${avg:,.2f} ({count} transactions)')
    
    # 6. Create Visualizations
    if visualize:
        create_validation_plots(df, validation_results, fraud_df)
    
    return df, chi2, p_value, validation_results

def create_validation_plots(df, results, fraud_df=None):
    """Create comprehensive validation visualizations"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Benford's Law Comparison
    ax1 = plt.subplot(2, 3, 1)
    digits = range(1, 10)
    width = 0.35
    ax1.bar([d - width/2 for d in digits], results['benford']['expected'], 
            width, label="Benford's Law", alpha=0.7, color='red')
    ax1.bar([d + width/2 for d in digits], results['benford']['observed'], 
            width, label='Observed', alpha=0.7, color='blue')
    ax1.set_xlabel('First Digit')
    ax1.set_ylabel('Proportion')
    ax1.set_title(f"Benford's Law Test (p={results['benford']['p_value']:.4f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Amount Distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(df['Amount'], bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Amount ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Transaction Amount Distribution')
    ax2.set_xlim(0, df['Amount'].quantile(0.95))
    
    # 3. Hourly Pattern
    ax3 = plt.subplot(2, 3, 3)
    hourly = df.groupby('Hour').size()
    ax3.bar(hourly.index, hourly.values, color='steelblue')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Transaction Count')
    ax3.set_title('Hourly Transaction Pattern')
    ax3.axvspan(9, 17, alpha=0.2, color='green', label='Market Hours')
    ax3.legend()
    
    # 4. Merchant Distribution
    ax4 = plt.subplot(2, 3, 4)
    merchant_counts = df['Merchant'].value_counts().sort_index()
    ax4.bar(merchant_counts.index, merchant_counts.values, color='coral')
    ax4.set_xlabel('Merchant')
    ax4.set_ylabel('Transaction Count')
    ax4.set_title('Transactions by Merchant')
    
    # 5. Location Distribution
    ax5 = plt.subplot(2, 3, 5)
    location_counts = df['Location'].value_counts()
    ax5.pie(location_counts.values, labels=location_counts.index, autopct='%1.1f%%')
    ax5.set_title('Transaction Distribution by Location')
    
    # 6. Fraud Pattern Distribution (if available)
    ax6 = plt.subplot(2, 3, 6)
    if fraud_df is not None and len(fraud_df) > 0:
        fraud_types = fraud_df['type'].value_counts()
        colors = plt.cm.Set3(range(len(fraud_types)))
        ax6.bar(range(len(fraud_types)), fraud_types.values, color=colors)
        ax6.set_xticks(range(len(fraud_types)))
        ax6.set_xticklabels(fraud_types.index, rotation=45, ha='right')
        ax6.set_ylabel('Count')
        ax6.set_title(f'Fraud Pattern Types (n={len(fraud_df)})')
    else:
        ax6.text(0.5, 0.5, 'No fraud data available', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Fraud Pattern Distribution')
    
    plt.suptitle('Enhanced Dataset Validation Report', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('./data', exist_ok=True)
    plt.savefig('./data/validation_report.png', dpi=100, bbox_inches='tight')
    print(f'\n📊 Validation plots saved to: ./data/validation_report.png')
    
    plt.show()

def generate_validation_report(results):
    """Generate a text report of validation results"""
    
    report = []
    report.append("="*60)
    report.append("ENHANCED DATASET VALIDATION REPORT")
    report.append("="*60)
    
    # Benford's Law Section
    report.append("\n1. BENFORD'S LAW COMPLIANCE")
    report.append("-"*30)
    if results['benford']['compliant']:
        report.append("✅ PASSED - Dataset follows natural distribution")
    else:
        report.append("⚠️  WARNING - Dataset shows signs of manipulation")
    report.append(f"Chi-squared: {results['benford']['chi_squared']:.2f}")
    report.append(f"P-value: {results['benford']['p_value']:.4f}")
    
    # Statistics Section
    report.append("\n2. DATASET STATISTICS")
    report.append("-"*30)
    report.append(f"Total transactions: {results['statistics']['total_transactions']:,}")
    report.append(f"Amount range: ${results['statistics']['amount_min']:.2f} - ${results['statistics']['amount_max']:.2f}")
    report.append(f"Average: ${results['statistics']['amount_mean']:.2f}")
    report.append(f"Median: ${results['statistics']['amount_median']:.2f}")
    
    # Temporal Section
    report.append("\n3. TEMPORAL PATTERNS")
    report.append("-"*30)
    report.append(f"Market hours concentration: {results['temporal']['market_concentration']:.1%}")
    report.append(f"Peak activity: {results['temporal']['peak_hour']}:00")
    
    # Fraud Section (if available)
    if 'fraud' in results:
        report.append("\n4. FRAUD PATTERNS")
        report.append("-"*30)
        report.append(f"Total fraud cases: {results['fraud']['total_patterns']:,}")
        report.append(f"Fraud rate: {results['fraud']['fraud_rate']*100:.2f}%")
        report.append("Type distribution:")
        for ftype, count in results['fraud']['type_distribution'].items():
            report.append(f"  - {ftype}: {count}")
    
    report.append("\n" + "="*60)
    report.append("END OF REPORT")
    report.append("="*60)
    
    # Save report
    report_text = "\n".join(report)
    with open('./data/validation_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f'\n📄 Text report saved to: ./data/validation_report.txt')
    
    return report_text

if __name__ == "__main__":
    # Run validation with visualizations
    print("="*60)
    print("RUNNING ENHANCED DATASET VALIDATION")
    print("="*60)
    
    enhanced_data, chi2_stat, benford_p, validation_results = validate_enhanced_dataset(visualize=True)
    
    # Generate text report
    report = generate_validation_report(validation_results)
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - validation_report.png (visualization)")
    print("  - validation_report.txt (text report)")