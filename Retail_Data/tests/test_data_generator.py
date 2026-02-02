"""Tests for RetailDataGenerator."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retail_data_generator import RetailDataGenerator


class TestRetailDataGeneratorInit:
    """Tests for RetailDataGenerator initialization."""

    def test_init_with_default_seed(self):
        """Test initialization with default seed."""
        generator = RetailDataGenerator()
        assert generator.random_seed == 42

    def test_init_with_custom_seed(self):
        """Test initialization with custom seed."""
        generator = RetailDataGenerator(random_seed=123)
        assert generator.random_seed == 123

    def test_regional_profiles_exist(self):
        """Test that all regional profiles are defined."""
        generator = RetailDataGenerator()
        expected_regions = ['West', 'North', 'East', 'South']
        for region in expected_regions:
            assert region in generator.regional_profiles
            assert 'income_multiplier' in generator.regional_profiles[region]
            assert 'urban_ratio' in generator.regional_profiles[region]
            assert 'online_preference' in generator.regional_profiles[region]

    def test_category_profiles_exist(self):
        """Test that all category profiles are defined."""
        generator = RetailDataGenerator()
        expected_categories = ['Electronics', 'Clothing', 'Grocery', 'Beauty', 'Furniture']
        for category in expected_categories:
            assert category in generator.category_profiles
            assert 'base_price_range' in generator.category_profiles[category]
            assert 'seasonal_multipliers' in generator.category_profiles[category]

    def test_age_group_profiles_exist(self):
        """Test that all age group profiles are defined."""
        generator = RetailDataGenerator()
        expected_groups = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        for group in expected_groups:
            assert group in generator.age_group_profiles
            assert 'spending_multiplier' in generator.age_group_profiles[group]


class TestReproducibility:
    """Tests for deterministic output with same seed."""

    def test_same_seed_produces_same_data(self):
        """Test that the same seed produces identical data."""
        gen1 = RetailDataGenerator(random_seed=42)
        gen2 = RetailDataGenerator(random_seed=42)

        df1 = gen1.generate_retail_dataset(num_transactions=100)
        df2 = gen2.generate_retail_dataset(num_transactions=100)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        gen1 = RetailDataGenerator(random_seed=42)
        gen2 = RetailDataGenerator(random_seed=123)

        df1 = gen1.generate_retail_dataset(num_transactions=100)
        df2 = gen2.generate_retail_dataset(num_transactions=100)

        # Should not be equal
        assert not df1.equals(df2)


class TestDatasetGeneration:
    """Tests for dataset generation functionality."""

    def test_generates_correct_number_of_transactions(self):
        """Test that the correct number of transactions is generated."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=500)
        assert len(df) == 500

    def test_all_columns_present(self):
        """Test that all expected columns are present."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=100)

        expected_columns = [
            'Transaction_ID', 'Customer_ID', 'Gender', 'Age', 'Category',
            'Quantity', 'Unit_Price', 'Discount', 'Date', 'Store_Region',
            'Online_Or_Offline', 'Payment_Method', 'Total_Amount'
        ]

        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_null_values(self):
        """Test that generated data has no null values."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=500)
        assert df.isnull().sum().sum() == 0

    def test_transaction_ids_unique(self):
        """Test that all transaction IDs are unique."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=1000)
        assert df['Transaction_ID'].nunique() == len(df)

    def test_dates_within_range(self):
        """Test that all dates are within specified range."""
        generator = RetailDataGenerator(random_seed=42)
        start = datetime(2022, 6, 1)
        end = datetime(2023, 6, 1)
        df = generator.generate_retail_dataset(
            num_transactions=500,
            start_date=start,
            end_date=end
        )

        df['Date'] = pd.to_datetime(df['Date'])
        assert df['Date'].min() >= start
        assert df['Date'].max() <= end

    def test_valid_categories(self):
        """Test that all categories are valid."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=500)

        valid_categories = ['Electronics', 'Clothing', 'Grocery', 'Beauty', 'Furniture']
        assert df['Category'].isin(valid_categories).all()

    def test_valid_regions(self):
        """Test that all regions are valid."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=500)

        valid_regions = ['West', 'North', 'East', 'South']
        assert df['Store_Region'].isin(valid_regions).all()

    def test_valid_genders(self):
        """Test that all genders are valid."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=500)

        valid_genders = ['Male', 'Female']
        assert df['Gender'].isin(valid_genders).all()

    def test_valid_channels(self):
        """Test that all channels are valid."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=500)

        valid_channels = ['Online', 'Offline']
        assert df['Online_Or_Offline'].isin(valid_channels).all()

    def test_valid_payment_methods(self):
        """Test that all payment methods are valid."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=500)

        valid_methods = ['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet']
        assert df['Payment_Method'].isin(valid_methods).all()

    def test_age_range(self):
        """Test that ages are within valid range."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=500)

        assert df['Age'].min() >= 18
        assert df['Age'].max() <= 85

    def test_positive_amounts(self):
        """Test that all amounts are positive."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=500)

        assert (df['Unit_Price'] > 0).all()
        assert (df['Quantity'] > 0).all()
        assert (df['Total_Amount'] > 0).all()

    def test_discount_range(self):
        """Test that discounts are within valid range."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=500)

        assert (df['Discount'] >= 0).all()
        assert (df['Discount'] <= 0.5).all()

    def test_total_amount_calculation(self):
        """Test that Total_Amount is calculated correctly."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=100)

        # Calculate expected total
        expected = df['Unit_Price'] * df['Quantity'] * (1 - df['Discount'])
        expected = expected.round(2)

        # Compare (allowing small floating point differences)
        np.testing.assert_array_almost_equal(df['Total_Amount'].values, expected.values, decimal=2)


class TestRegionalPatterns:
    """Tests for regional differentiation patterns."""

    def test_regional_income_differences(self):
        """Test that regional income differences are reflected in spending."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=5000)

        regional_means = df.groupby('Store_Region')['Total_Amount'].mean()

        # West should have higher spending than South
        assert regional_means['West'] > regional_means['South']

    def test_all_regions_represented(self):
        """Test that all regions are represented in the data."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=1000)

        regions_in_data = set(df['Store_Region'].unique())
        expected_regions = {'West', 'North', 'East', 'South'}

        assert regions_in_data == expected_regions


class TestCustomerIdBehavior:
    """Tests for customer ID generation patterns."""

    def test_customer_id_format(self):
        """Test that customer IDs follow expected format."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=500)

        # All customer IDs should start with 'CUST-'
        assert df['Customer_ID'].str.startswith('CUST-').all()

    def test_online_repeat_customers_exist(self):
        """Test that online channel has repeat customers."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=2000)

        online = df[df['Online_Or_Offline'] == 'Online']
        online_customer_counts = online['Customer_ID'].value_counts()

        # Should have some repeat customers (count > 1)
        repeat_customers = (online_customer_counts > 1).sum()
        assert repeat_customers > 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimum_transactions(self):
        """Test generating minimum valid transactions."""
        generator = RetailDataGenerator(random_seed=42)
        df = generator.generate_retail_dataset(num_transactions=100)
        assert len(df) == 100

    def test_invalid_transaction_count_raises(self):
        """Test that invalid transaction count raises ValueError."""
        generator = RetailDataGenerator(random_seed=42)
        with pytest.raises(ValueError):
            generator.generate_retail_dataset(num_transactions=50)  # Too few

    def test_single_day_date_range(self):
        """Test generation with single day date range."""
        generator = RetailDataGenerator(random_seed=42)
        single_day = datetime(2023, 1, 15)
        df = generator.generate_retail_dataset(
            num_transactions=100,
            start_date=single_day,
            end_date=single_day
        )

        df['Date'] = pd.to_datetime(df['Date'])
        assert (df['Date'].dt.date == single_day.date()).all()


class TestStudentDatasetGeneration:
    """Tests for student dataset generation."""

    def test_generates_multiple_datasets(self, temp_dir):
        """Test generating multiple student datasets."""
        generator = RetailDataGenerator(random_seed=42)

        # Temporarily change to temp directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)

        try:
            result = generator.generate_student_datasets(
                num_datasets=3,
                num_transactions=100
            )

            assert len(result['data_files']) == 3
            assert len(result['mapping']) == 3

            # Check files exist
            for file_path in result['data_files']:
                assert os.path.exists(file_path)
        finally:
            os.chdir(original_dir)

    def test_student_datasets_are_different(self, temp_dir):
        """Test that each student dataset is unique."""
        generator = RetailDataGenerator(random_seed=42)

        original_dir = os.getcwd()
        os.chdir(temp_dir)

        try:
            result = generator.generate_student_datasets(
                num_datasets=2,
                num_transactions=100
            )

            df1 = pd.read_csv(result['data_files'][0])
            df2 = pd.read_csv(result['data_files'][1])

            # Datasets should be different
            assert not df1.equals(df2)
        finally:
            os.chdir(original_dir)
