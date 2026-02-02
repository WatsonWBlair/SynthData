"""Tests for validate_retail_data module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validate_retail_data import (
    validate_regional_patterns,
    validate_seasonal_patterns,
    validate_demographic_patterns,
    validate_customer_id_behavior,
    generate_validation_report
)


class TestValidateRegionalPatterns:
    """Tests for regional pattern validation."""

    def test_returns_dict(self, sample_retail_data):
        """Test that function returns a dictionary."""
        result = validate_regional_patterns(sample_retail_data)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, sample_retail_data):
        """Test that result contains required keys."""
        result = validate_regional_patterns(sample_retail_data)

        required_keys = ['regional_means', 'regional_counts', 'anova_f_stat',
                        'anova_p_value', 'significant_difference']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_regional_means_calculated(self, sample_retail_data):
        """Test that regional means are calculated correctly."""
        result = validate_regional_patterns(sample_retail_data)

        # Calculate expected means manually
        expected_means = sample_retail_data.groupby('Store_Region')['Total_Amount'].mean()

        for region in expected_means.index:
            assert region in result['regional_means']
            np.testing.assert_almost_equal(
                result['regional_means'][region],
                expected_means[region],
                decimal=2
            )

    def test_detects_significant_regional_differences(self, regional_test_data):
        """Test detection of significant regional differences."""
        result = validate_regional_patterns(regional_test_data)

        # With intentionally different regional spending, should be significant
        assert result['significant_difference'] == True  # Use == for numpy bool compatibility
        assert result['anova_p_value'] < 0.05


class TestValidateSeasonalPatterns:
    """Tests for seasonal pattern validation."""

    def test_returns_dict(self, sample_retail_data):
        """Test that function returns a dictionary."""
        result = validate_seasonal_patterns(sample_retail_data)
        assert isinstance(result, dict)

    def test_quarterly_data_present(self, seasonal_test_data):
        """Test that quarterly data is calculated."""
        result = validate_seasonal_patterns(seasonal_test_data)
        assert 'quarterly_data' in result

    def test_weekend_multiplier_calculated(self, sample_retail_data):
        """Test that weekend multiplier is calculated."""
        result = validate_seasonal_patterns(sample_retail_data)
        assert 'weekend_multiplier' in result
        assert isinstance(result['weekend_multiplier'], float)


class TestValidateDemographicPatterns:
    """Tests for demographic pattern validation."""

    def test_returns_dict(self, sample_retail_data):
        """Test that function returns a dictionary."""
        result = validate_demographic_patterns(sample_retail_data)
        assert isinstance(result, dict)

    def test_age_group_spending_calculated(self, sample_retail_data):
        """Test that age group spending is calculated."""
        result = validate_demographic_patterns(sample_retail_data)
        assert 'age_group_spending' in result

    def test_gender_category_analysis(self, sample_retail_data):
        """Test gender-category analysis."""
        # Create data with strong gender-Beauty correlation
        data = sample_retail_data.copy()
        # Increase Beauty purchases for females
        beauty_mask = data['Category'] == 'Beauty'
        female_mask = data['Gender'] == 'Female'
        data.loc[beauty_mask & female_mask, 'Total_Amount'] *= 1.5

        result = validate_demographic_patterns(data)
        assert 'beauty_gender_diff' in result


class TestValidateCustomerIdBehavior:
    """Tests for customer ID behavior validation."""

    def test_returns_dict(self, sample_retail_data):
        """Test that function returns a dictionary."""
        result = validate_customer_id_behavior(sample_retail_data)
        assert isinstance(result, dict)

    def test_online_metrics_present(self, sample_retail_data):
        """Test that online customer metrics are present."""
        result = validate_customer_id_behavior(sample_retail_data)

        assert 'online_unique_customers' in result
        assert 'online_repeat_customers' in result
        assert 'online_repeat_rate' in result

    def test_offline_metrics_present(self, sample_retail_data):
        """Test that offline customer metrics are present."""
        result = validate_customer_id_behavior(sample_retail_data)

        assert 'offline_unique_customers' in result
        assert 'offline_single_visit' in result
        assert 'offline_transient_rate' in result

    def test_repeat_rate_range(self, sample_retail_data):
        """Test that repeat rates are between 0 and 1."""
        result = validate_customer_id_behavior(sample_retail_data)

        assert 0 <= result['online_repeat_rate'] <= 1
        assert 0 <= result['offline_transient_rate'] <= 1

    def test_detects_online_repeat_pattern(self, customer_id_test_data):
        """Test detection of online repeat customer pattern."""
        result = validate_customer_id_behavior(customer_id_test_data)

        # Online should have more repeat customers
        assert result['online_repeat_rate'] > 0.3

    def test_detects_offline_transient_pattern(self, customer_id_test_data):
        """Test detection of offline transient pattern."""
        result = validate_customer_id_behavior(customer_id_test_data)

        # Offline should have more transient customers
        assert result['offline_transient_rate'] > 0.4


class TestGenerateValidationReport:
    """Tests for validation report generation."""

    def test_generates_text_report(self, sample_retail_data):
        """Test that a text report is generated."""
        # Create mock results
        results = {
            'regional': {
                'anova_p_value': 0.001,
                'significant_difference': True,
                'regional_means': {'West': 300, 'North': 280, 'East': 250, 'South': 200}
            },
            'seasonal': {
                'electronics_q4_spike': 25.0,
                'black_friday_multiplier': 2.0,
                'weekend_multiplier': 1.2
            },
            'demographic': {
                'beauty_gender_diff': 45.0,
                'electronics_age_corr': -0.15
            },
            'customer_id': {
                'online_repeat_rate': 0.55,
                'offline_transient_rate': 0.65,
                'online_avg_transactions': 2.5,
                'offline_avg_transactions': 1.3
            }
        }

        report = generate_validation_report(results)

        assert isinstance(report, str)
        assert 'VALIDATION REPORT' in report
        assert 'REGIONAL PATTERNS' in report
        assert 'SEASONAL PATTERNS' in report

    def test_report_saves_to_file(self, sample_retail_data, temp_dir):
        """Test that report can be saved to file."""
        results = {
            'regional': {'anova_p_value': 0.001, 'significant_difference': True, 'regional_means': {}},
            'seasonal': {},
            'demographic': {},
            'customer_id': {'online_repeat_rate': 0.5, 'offline_transient_rate': 0.6}
        }

        output_path = os.path.join(temp_dir, 'test_report.txt')
        generate_validation_report(results, output_path)

        assert os.path.exists(output_path)
        with open(output_path, 'r') as f:
            content = f.read()
            assert 'VALIDATION REPORT' in content


class TestIntegration:
    """Integration tests for validation module."""

    def test_full_validation_pipeline(self, sample_retail_data):
        """Test running all validations on sample data."""
        # Should run without errors
        regional = validate_regional_patterns(sample_retail_data)
        seasonal = validate_seasonal_patterns(sample_retail_data)
        demographic = validate_demographic_patterns(sample_retail_data)
        customer_id = validate_customer_id_behavior(sample_retail_data)

        all_results = {
            'regional': regional,
            'seasonal': seasonal,
            'demographic': demographic,
            'customer_id': customer_id
        }

        # Generate report
        report = generate_validation_report(all_results)
        assert len(report) > 0

    def test_handles_empty_channel(self, sample_retail_data):
        """Test handling data with only one channel."""
        # Create data with only online transactions
        online_only = sample_retail_data[sample_retail_data['Online_Or_Offline'] == 'Online'].copy()

        if len(online_only) > 0:
            result = validate_customer_id_behavior(online_only)
            assert 'online_unique_customers' in result
