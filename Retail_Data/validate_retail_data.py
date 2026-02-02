#!/usr/bin/env python3
"""
Retail Data Validation Script with Visualizations
==================================================

Validates statistical patterns and generates comprehensive reports:
1. Regional differentiation (spend, online preference)
2. Seasonal patterns (quarterly, holidays)
3. Demographic patterns (age-category, gender-category correlations)
4. Customer ID behavior (online persistent vs in-store transient)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from typing import Dict, Tuple, Optional


def validate_regional_patterns(df: pd.DataFrame) -> Dict:
    """
    Validate regional spend differences.

    Expected: West/North > East > South in average spending
    Test: ANOVA with p < 0.05

    Args:
        df: Retail transaction DataFrame

    Returns:
        Dictionary with validation results
    """
    print("\n=== Regional Pattern Validation ===")

    # Group by region and calculate statistics
    regional_stats = df.groupby('Store_Region')['Total_Amount'].agg(['mean', 'std', 'count', 'sum'])
    regional_stats['avg_transaction'] = regional_stats['sum'] / regional_stats['count']

    print("\nRegional Spending Statistics:")
    for region in ['West', 'North', 'East', 'South']:
        if region in regional_stats.index:
            row = regional_stats.loc[region]
            print(f"  {region}: ${row['mean']:.2f} avg (n={int(row['count']):,})")

    # ANOVA test for regional differences
    regions = df['Store_Region'].unique()
    region_groups = [df[df['Store_Region'] == r]['Total_Amount'].values for r in regions]
    f_stat, p_value = stats.f_oneway(*region_groups)

    print(f"\nANOVA Test:")
    print(f"  F-statistic: {f_stat:.2f}")
    print(f"  P-value: {p_value:.6f}")

    # Check expected ordering
    means = regional_stats['mean'].sort_values(ascending=False)
    expected_high = ['West', 'North']
    expected_low = ['South']

    ordering_valid = True
    for high_region in expected_high:
        for low_region in expected_low:
            if high_region in means.index and low_region in means.index:
                if means[high_region] <= means[low_region]:
                    ordering_valid = False

    result = {
        'regional_means': regional_stats['mean'].to_dict(),
        'regional_counts': regional_stats['count'].to_dict(),
        'anova_f_stat': float(f_stat),
        'anova_p_value': float(p_value),
        'significant_difference': p_value < 0.05,
        'ordering_valid': ordering_valid
    }

    if p_value < 0.05:
        print("  PASSED: Significant regional differences detected")
    else:
        print("  WARNING: No significant regional differences")

    # Online preference by region
    online_by_region = df.groupby('Store_Region')['Online_Or_Offline'].apply(
        lambda x: (x == 'Online').mean()
    ).to_dict()

    print("\nOnline Shopping Preference by Region:")
    for region in ['West', 'North', 'East', 'South']:
        if region in online_by_region:
            print(f"  {region}: {online_by_region[region]*100:.1f}%")

    result['online_preference'] = online_by_region

    return result


def validate_seasonal_patterns(df: pd.DataFrame) -> Dict:
    """
    Verify holiday spikes and quarterly patterns.

    Expected:
    - Q4 Electronics spike (+35% vs baseline)
    - Black Friday 2-2.5x normal week

    Args:
        df: Retail transaction DataFrame

    Returns:
        Dictionary with validation results
    """
    print("\n=== Seasonal Pattern Validation ===")

    # Convert date column if needed
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])

    # Add temporal columns
    df_temp = df.copy()
    df_temp['Quarter'] = df_temp['Date'].dt.quarter.map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
    df_temp['Month'] = df_temp['Date'].dt.month
    df_temp['Week'] = df_temp['Date'].dt.isocalendar().week
    df_temp['Year'] = df_temp['Date'].dt.year
    df_temp['DayOfWeek'] = df_temp['Date'].dt.dayofweek

    results = {}

    # Quarterly analysis by category
    print("\nQuarterly Sales by Category:")
    quarterly_category = df_temp.groupby(['Quarter', 'Category'])['Total_Amount'].sum().unstack()

    # Normalize to Q1 baseline
    if 'Q1' in quarterly_category.index:
        quarterly_normalized = quarterly_category.div(quarterly_category.loc['Q1'])
    else:
        quarterly_normalized = quarterly_category.div(quarterly_category.iloc[0])

    print("\nQuarterly Index (Q1 = 1.0):")
    for category in quarterly_normalized.columns:
        row = quarterly_normalized[category]
        quarters_str = []
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            if q in row.index:
                quarters_str.append(f"{q}={row[q]:.2f}")
            else:
                quarters_str.append(f"{q}=N/A")
        print(f"  {category}: {', '.join(quarters_str)}")

    # Check Electronics Q4 spike
    if 'Electronics' in quarterly_normalized.columns:
        electronics_q4 = quarterly_normalized.loc['Q4', 'Electronics'] if 'Q4' in quarterly_normalized.index else 1.0
        electronics_baseline = quarterly_normalized['Electronics'].mean()
        q4_spike = (electronics_q4 - electronics_baseline) / electronics_baseline * 100

        print(f"\nElectronics Q4 Spike: {q4_spike:+.1f}% vs average")
        results['electronics_q4_spike'] = float(q4_spike)
        results['electronics_q4_valid'] = q4_spike > 15  # At least 15% increase

    # Black Friday analysis (week 47-48, late November)
    black_friday_weeks = df_temp[df_temp['Week'].isin([47, 48])]
    normal_weeks = df_temp[~df_temp['Week'].isin([47, 48, 51, 52, 1])]  # Exclude holidays

    if len(black_friday_weeks) > 0 and len(normal_weeks) > 0:
        bf_daily_avg = black_friday_weeks['Total_Amount'].sum() / black_friday_weeks['Date'].nunique()
        normal_daily_avg = normal_weeks['Total_Amount'].sum() / normal_weeks['Date'].nunique()

        bf_multiplier = bf_daily_avg / normal_daily_avg if normal_daily_avg > 0 else 1.0

        print(f"\nBlack Friday Week Analysis:")
        print(f"  Black Friday period daily avg: ${bf_daily_avg:,.2f}")
        print(f"  Normal period daily avg: ${normal_daily_avg:,.2f}")
        print(f"  Multiplier: {bf_multiplier:.2f}x")

        results['black_friday_multiplier'] = float(bf_multiplier)
        results['black_friday_valid'] = bf_multiplier > 1.5

    # Weekend vs weekday analysis
    weekend = df_temp[df_temp['DayOfWeek'] >= 5]
    weekday = df_temp[df_temp['DayOfWeek'] < 5]

    weekend_avg = weekend['Total_Amount'].mean() if len(weekend) > 0 else 0
    weekday_avg = weekday['Total_Amount'].mean() if len(weekday) > 0 else 0

    weekend_multiplier = weekend_avg / weekday_avg if weekday_avg > 0 else 1.0

    print(f"\nWeekend vs Weekday:")
    print(f"  Weekend avg: ${weekend_avg:.2f}")
    print(f"  Weekday avg: ${weekday_avg:.2f}")
    print(f"  Weekend multiplier: {weekend_multiplier:.2f}x")

    results['weekend_multiplier'] = float(weekend_multiplier)
    results['quarterly_data'] = quarterly_normalized.to_dict()

    return results


def validate_demographic_patterns(df: pd.DataFrame) -> Dict:
    """
    Validate age and gender correlations with categories.

    Expected:
    - Age-Electronics: Negative correlation (r < -0.2)
    - Gender-Beauty: Female 50%+ higher

    Args:
        df: Retail transaction DataFrame

    Returns:
        Dictionary with validation results
    """
    print("\n=== Demographic Pattern Validation ===")

    results = {}

    # Age-Category Analysis
    print("\nAge-Category Correlations:")

    for category in df['Category'].unique():
        cat_data = df[df['Category'] == category]
        if len(cat_data) > 10:
            corr, p_val = stats.pearsonr(cat_data['Age'], cat_data['Total_Amount'])
            print(f"  {category}: r={corr:.3f} (p={p_val:.4f})")
            results[f'age_{category.lower()}_corr'] = float(corr)

    # Check Electronics-Age correlation
    electronics = df[df['Category'] == 'Electronics']
    if len(electronics) > 10:
        elec_corr, elec_p = stats.pearsonr(electronics['Age'], electronics['Total_Amount'])
        results['electronics_age_corr'] = float(elec_corr)
        results['electronics_age_valid'] = elec_corr < -0.1  # Negative correlation expected

    # Gender-Category Analysis
    print("\nGender-Category Spending:")
    gender_category = df.groupby(['Gender', 'Category'])['Total_Amount'].mean().unstack()

    for category in gender_category.columns:
        if 'Female' in gender_category.index and 'Male' in gender_category.index:
            female_avg = gender_category.loc['Female', category]
            male_avg = gender_category.loc['Male', category]
            diff_pct = (female_avg - male_avg) / male_avg * 100
            print(f"  {category}: Female ${female_avg:.2f}, Male ${male_avg:.2f} ({diff_pct:+.1f}%)")

    # Beauty gender difference
    if 'Beauty' in gender_category.columns:
        beauty_female = gender_category.loc['Female', 'Beauty'] if 'Female' in gender_category.index else 0
        beauty_male = gender_category.loc['Male', 'Beauty'] if 'Male' in gender_category.index else 0

        beauty_gender_diff = (beauty_female - beauty_male) / beauty_male * 100 if beauty_male > 0 else 0

        print(f"\nBeauty Gender Difference: {beauty_gender_diff:+.1f}%")
        results['beauty_gender_diff'] = float(beauty_gender_diff)
        results['beauty_gender_valid'] = beauty_gender_diff > 30  # Female at least 30% higher

    # Age group spending analysis
    print("\nSpending by Age Group:")
    age_bins = [18, 26, 36, 46, 56, 66, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']

    df_temp = df.copy()
    df_temp['Age_Group'] = pd.cut(df_temp['Age'], bins=age_bins, labels=age_labels, right=False)

    age_group_stats = df_temp.groupby('Age_Group')['Total_Amount'].mean()
    for age_group in age_labels:
        if age_group in age_group_stats.index:
            print(f"  {age_group}: ${age_group_stats[age_group]:.2f}")

    results['age_group_spending'] = age_group_stats.to_dict()

    # Verify 36-45 group spends most
    if '36-45' in age_group_stats.index:
        peak_age_group = age_group_stats.idxmax()
        results['peak_spending_age_valid'] = peak_age_group in ['36-45', '46-55']
        print(f"\nPeak spending age group: {peak_age_group}")

    return results


def validate_customer_id_behavior(df: pd.DataFrame) -> Dict:
    """
    Validate online persistent vs in-store transient ID patterns.

    Expected:
    - Online repeat rate: >50%
    - In-store new rate: ~70%

    Args:
        df: Retail transaction DataFrame

    Returns:
        Dictionary with validation results
    """
    print("\n=== Customer ID Behavior Validation ===")

    results = {}

    # Online customer analysis
    online = df[df['Online_Or_Offline'] == 'Online']
    online_customer_counts = online['Customer_ID'].value_counts()
    online_repeat_customers = (online_customer_counts > 1).sum()
    online_total_customers = len(online_customer_counts)
    online_repeat_rate = online_repeat_customers / online_total_customers if online_total_customers > 0 else 0

    print("\nOnline Customer Analysis:")
    print(f"  Total unique customers: {online_total_customers:,}")
    print(f"  Repeat customers (>1 purchase): {online_repeat_customers:,}")
    print(f"  Repeat rate: {online_repeat_rate*100:.1f}%")

    results['online_unique_customers'] = int(online_total_customers)
    results['online_repeat_customers'] = int(online_repeat_customers)
    results['online_repeat_rate'] = float(online_repeat_rate)
    results['online_repeat_valid'] = online_repeat_rate > 0.40  # At least 40% repeat

    # In-store customer analysis
    offline = df[df['Online_Or_Offline'] == 'Offline']
    offline_customer_counts = offline['Customer_ID'].value_counts()
    offline_single_visit = (offline_customer_counts == 1).sum()
    offline_total_customers = len(offline_customer_counts)
    offline_transient_rate = offline_single_visit / offline_total_customers if offline_total_customers > 0 else 0

    print("\nIn-Store Customer Analysis:")
    print(f"  Total unique customers: {offline_total_customers:,}")
    print(f"  Single-visit customers: {offline_single_visit:,}")
    print(f"  Transient rate: {offline_transient_rate*100:.1f}%")

    results['offline_unique_customers'] = int(offline_total_customers)
    results['offline_single_visit'] = int(offline_single_visit)
    results['offline_transient_rate'] = float(offline_transient_rate)
    results['offline_transient_valid'] = offline_transient_rate > 0.50  # At least 50% transient

    # Average transactions per customer
    online_avg_txn = online_customer_counts.mean() if len(online_customer_counts) > 0 else 0
    offline_avg_txn = offline_customer_counts.mean() if len(offline_customer_counts) > 0 else 0

    print(f"\nAverage Transactions per Customer:")
    print(f"  Online: {online_avg_txn:.2f}")
    print(f"  In-store: {offline_avg_txn:.2f}")

    results['online_avg_transactions'] = float(online_avg_txn)
    results['offline_avg_transactions'] = float(offline_avg_txn)

    return results


def create_validation_plots(df: pd.DataFrame, validation_results: Dict,
                           output_path: Optional[str] = None) -> None:
    """
    Create 6-panel visualization dashboard.

    Panels:
    1. Regional spend comparison
    2. Category by gender
    3. Quarterly trends
    4. Age distribution by category
    5. Online vs Offline breakdown
    6. Customer repeat rates

    Args:
        df: Retail transaction DataFrame
        validation_results: Dictionary with all validation results
        output_path: Optional path to save figure
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    fig = plt.figure(figsize=(20, 14))

    # Convert date if needed
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])

    # 1. Regional Spend Comparison
    ax1 = plt.subplot(2, 3, 1)
    regional_means = df.groupby('Store_Region')['Total_Amount'].mean().sort_values(ascending=False)
    colors = ['#2ecc71' if r in ['West', 'North'] else '#e74c3c' if r == 'South' else '#3498db'
              for r in regional_means.index]
    bars = ax1.bar(regional_means.index, regional_means.values, color=colors)
    ax1.set_xlabel('Region')
    ax1.set_ylabel('Average Transaction ($)')
    ax1.set_title('Average Spending by Region')
    # Add value labels
    for bar, val in zip(bars, regional_means.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'${val:.0f}', ha='center', va='bottom', fontsize=10)

    # 2. Category by Gender
    ax2 = plt.subplot(2, 3, 2)
    gender_category = df.groupby(['Gender', 'Category'])['Total_Amount'].mean().unstack()
    gender_category.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_xlabel('Gender')
    ax2.set_ylabel('Average Transaction ($)')
    ax2.set_title('Category Spending by Gender')
    ax2.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

    # 3. Quarterly Trends
    ax3 = plt.subplot(2, 3, 3)
    df_temp = df.copy()
    df_temp['Quarter'] = df_temp['Date'].dt.quarter.map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
    quarterly = df_temp.groupby(['Quarter', 'Category'])['Total_Amount'].sum().unstack()
    # Normalize to show relative patterns
    quarterly_norm = quarterly.div(quarterly.mean())
    quarterly_norm.plot(kind='line', marker='o', ax=ax3)
    ax3.set_xlabel('Quarter')
    ax3.set_ylabel('Relative Sales (1.0 = Average)')
    ax3.set_title('Quarterly Sales Patterns by Category')
    ax3.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # 4. Age Distribution by Category (Heatmap)
    ax4 = plt.subplot(2, 3, 4)
    age_bins = [18, 26, 36, 46, 56, 66, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    df_temp['Age_Group'] = pd.cut(df_temp['Age'], bins=age_bins, labels=age_labels, right=False)
    age_category = df_temp.groupby(['Age_Group', 'Category'])['Total_Amount'].mean().unstack()
    # Normalize by category to show relative preferences
    age_category_norm = age_category.div(age_category.mean())
    sns.heatmap(age_category_norm, annot=True, fmt='.2f', cmap='RdYlGn',
                center=1.0, ax=ax4, cbar_kws={'label': 'Relative Spending'})
    ax4.set_xlabel('Category')
    ax4.set_ylabel('Age Group')
    ax4.set_title('Spending Index by Age Group & Category')

    # 5. Online vs Offline Breakdown
    ax5 = plt.subplot(2, 3, 5)
    channel_region = df.groupby(['Store_Region', 'Online_Or_Offline']).size().unstack()
    channel_region_pct = channel_region.div(channel_region.sum(axis=1), axis=0) * 100
    channel_region_pct.plot(kind='bar', stacked=True, ax=ax5, color=['#3498db', '#e74c3c'])
    ax5.set_xlabel('Region')
    ax5.set_ylabel('Percentage')
    ax5.set_title('Online vs Offline by Region')
    ax5.legend(title='Channel')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)
    # Add percentage labels
    for container in ax5.containers:
        ax5.bar_label(container, fmt='%.0f%%', label_type='center', fontsize=9)

    # 6. Customer Repeat Rates
    ax6 = plt.subplot(2, 3, 6)
    online = df[df['Online_Or_Offline'] == 'Online']
    offline = df[df['Online_Or_Offline'] == 'Offline']

    online_counts = online['Customer_ID'].value_counts()
    offline_counts = offline['Customer_ID'].value_counts()

    online_repeat = (online_counts > 1).sum() / len(online_counts) * 100 if len(online_counts) > 0 else 0
    offline_repeat = (offline_counts > 1).sum() / len(offline_counts) * 100 if len(offline_counts) > 0 else 0

    channels = ['Online', 'In-Store']
    repeat_rates = [online_repeat, offline_repeat]
    single_rates = [100 - online_repeat, 100 - offline_repeat]

    x = np.arange(len(channels))
    width = 0.35
    bars1 = ax6.bar(x - width/2, repeat_rates, width, label='Repeat Customers', color='#2ecc71')
    bars2 = ax6.bar(x + width/2, single_rates, width, label='Single Visit', color='#e74c3c')

    ax6.set_ylabel('Percentage')
    ax6.set_title('Customer Behavior by Channel')
    ax6.set_xticks(x)
    ax6.set_xticklabels(channels)
    ax6.legend()
    ax6.set_ylim(0, 100)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2, height + 2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Retail Data Validation Dashboard', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save plot
    if output_path is None:
        os.makedirs('./data', exist_ok=True)
        output_path = './data/validation_report.png'

    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\nValidation plots saved to: {output_path}")

    plt.show()


def validate_retail_dataset(data_path: str = './data/retail_data.csv',
                           visualize: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Comprehensive validation of generated retail dataset.

    Args:
        data_path: Path to the retail data CSV
        visualize: Whether to create visualization plots

    Returns:
        Tuple of (DataFrame, validation_results dictionary)
    """
    # Load dataset
    print("="*60)
    print("RETAIL DATA VALIDATION")
    print("="*60)

    df = pd.read_csv(data_path)
    print(f"\nDataset loaded: {len(df):,} transactions")
    print(f"Columns: {df.columns.tolist()}")

    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'])

    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Total Amount range: ${df['Total_Amount'].min():.2f} - ${df['Total_Amount'].max():.2f}")
    print(f"Average transaction: ${df['Total_Amount'].mean():.2f}")
    print(f"Median transaction: ${df['Total_Amount'].median():.2f}")

    # Run all validations
    validation_results = {}

    validation_results['regional'] = validate_regional_patterns(df)
    validation_results['seasonal'] = validate_seasonal_patterns(df)
    validation_results['demographic'] = validate_demographic_patterns(df)
    validation_results['customer_id'] = validate_customer_id_behavior(df)

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    checks = [
        ('Regional Differences', validation_results['regional'].get('significant_difference', False)),
        ('Regional Ordering', validation_results['regional'].get('ordering_valid', False)),
        ('Electronics Q4 Spike', validation_results['seasonal'].get('electronics_q4_valid', False)),
        ('Beauty Gender Diff', validation_results['demographic'].get('beauty_gender_valid', False)),
        ('Online Repeat Rate', validation_results['customer_id'].get('online_repeat_valid', False)),
        ('Offline Transient Rate', validation_results['customer_id'].get('offline_transient_valid', False)),
    ]

    passed = sum(1 for _, valid in checks if valid)
    total = len(checks)

    print(f"\nValidation Checks: {passed}/{total} passed")
    for name, valid in checks:
        status = "PASS" if valid else "FAIL"
        print(f"  [{status}] {name}")

    # Create visualizations
    if visualize:
        create_validation_plots(df, validation_results)

    return df, validation_results


def generate_validation_report(results: Dict, output_path: Optional[str] = None) -> str:
    """
    Generate a text report of validation results.

    Args:
        results: Dictionary with all validation results
        output_path: Optional path to save report

    Returns:
        Report text string
    """
    report = []
    report.append("="*60)
    report.append("RETAIL DATA VALIDATION REPORT")
    report.append("="*60)

    # Regional Section
    report.append("\n1. REGIONAL PATTERNS")
    report.append("-"*30)
    if 'regional' in results:
        r = results['regional']
        p_val = r.get('anova_p_value')
        report.append(f"ANOVA p-value: {p_val:.6f}" if p_val is not None else "ANOVA p-value: N/A")
        report.append(f"Significant difference: {'Yes' if r.get('significant_difference') else 'No'}")
        report.append("Regional means:")
        for region, mean in r.get('regional_means', {}).items():
            report.append(f"  {region}: ${mean:.2f}")

    # Seasonal Section
    report.append("\n2. SEASONAL PATTERNS")
    report.append("-"*30)
    if 'seasonal' in results:
        s = results['seasonal']
        q4_spike = s.get('electronics_q4_spike')
        bf_mult = s.get('black_friday_multiplier')
        wk_mult = s.get('weekend_multiplier')
        report.append(f"Electronics Q4 spike: {q4_spike:.1f}%" if q4_spike is not None else "Electronics Q4 spike: N/A")
        report.append(f"Black Friday multiplier: {bf_mult:.2f}x" if bf_mult is not None else "Black Friday multiplier: N/A")
        report.append(f"Weekend multiplier: {wk_mult:.2f}x" if wk_mult is not None else "Weekend multiplier: N/A")

    # Demographic Section
    report.append("\n3. DEMOGRAPHIC PATTERNS")
    report.append("-"*30)
    if 'demographic' in results:
        d = results['demographic']
        beauty_diff = d.get('beauty_gender_diff')
        elec_corr = d.get('electronics_age_corr')
        report.append(f"Beauty gender difference: {beauty_diff:.1f}%" if beauty_diff is not None else "Beauty gender difference: N/A")
        report.append(f"Electronics-Age correlation: {elec_corr:.3f}" if elec_corr is not None else "Electronics-Age correlation: N/A")

    # Customer ID Section
    report.append("\n4. CUSTOMER ID BEHAVIOR")
    report.append("-"*30)
    if 'customer_id' in results:
        c = results['customer_id']
        online_rate = c.get('online_repeat_rate', 0)
        offline_rate = c.get('offline_transient_rate', 0)
        online_avg = c.get('online_avg_transactions')
        offline_avg = c.get('offline_avg_transactions')
        report.append(f"Online repeat rate: {online_rate*100:.1f}%")
        report.append(f"Offline transient rate: {offline_rate*100:.1f}%")
        report.append(f"Online avg transactions: {online_avg:.2f}" if online_avg is not None else "Online avg transactions: N/A")
        report.append(f"Offline avg transactions: {offline_avg:.2f}" if offline_avg is not None else "Offline avg transactions: N/A")

    report.append("\n" + "="*60)
    report.append("END OF REPORT")
    report.append("="*60)

    report_text = "\n".join(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"\nText report saved to: {output_path}")

    return report_text


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(
        description='Validate retail data patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Validate default data file
  python validate_retail_data.py

  # Validate specific file
  python validate_retail_data.py --input ./data/team_001_data.csv

  # Generate text report
  python validate_retail_data.py --report ./data/validation_report.txt
        ''')

    parser.add_argument('--input', type=str, default='./data/retail_data.csv',
                       help='Path to retail data CSV (default: ./data/retail_data.csv)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--report', type=str, default=None,
                       help='Path to save text report')

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        print("Run retail_data_generator.py first to generate data.")
        return

    # Run validation
    df, results = validate_retail_dataset(
        data_path=args.input,
        visualize=not args.no_visualize
    )

    # Generate text report if requested
    if args.report:
        generate_validation_report(results, args.report)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
