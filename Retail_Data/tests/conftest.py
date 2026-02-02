"""Shared fixtures for retail data test suite."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from datetime import datetime


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_retail_data():
    """Create sample retail transaction data for testing."""
    np.random.seed(42)
    n_transactions = 400  # Increased to cover full year

    data = {
        'Transaction_ID': [f'TXN-{i:05d}' for i in range(n_transactions)],
        'Customer_ID': [f'CUST-{np.random.randint(0, 50):04d}' for _ in range(n_transactions)],
        'Gender': np.random.choice(['Male', 'Female'], n_transactions, p=[0.48, 0.52]),
        'Age': np.random.randint(18, 75, n_transactions),
        'Category': np.random.choice(
            ['Electronics', 'Clothing', 'Grocery', 'Beauty', 'Furniture'],
            n_transactions, p=[0.2, 0.25, 0.25, 0.15, 0.15]
        ),
        'Quantity': np.random.randint(1, 10, n_transactions),
        'Unit_Price': np.random.uniform(10, 500, n_transactions).round(2),
        'Discount': np.random.uniform(0, 0.3, n_transactions).round(2),
        'Date': pd.date_range('2022-01-01', periods=n_transactions, freq='D'),  # Spans full year
        'Store_Region': np.random.choice(
            ['West', 'North', 'East', 'South'],
            n_transactions, p=[0.25, 0.25, 0.30, 0.20]
        ),
        'Online_Or_Offline': np.random.choice(
            ['Online', 'Offline'],
            n_transactions, p=[0.45, 0.55]
        ),
        'Payment_Method': np.random.choice(
            ['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet'],
            n_transactions, p=[0.30, 0.25, 0.25, 0.20]
        ),
    }

    df = pd.DataFrame(data)
    # Calculate Total_Amount
    df['Total_Amount'] = (df['Unit_Price'] * df['Quantity'] * (1 - df['Discount'])).round(2)

    return df


@pytest.fixture
def sample_predictions(sample_retail_data):
    """Create sample prediction data for testing grading tools."""
    np.random.seed(42)
    actual = sample_retail_data['Total_Amount'].values

    # Generate predictions with some noise
    noise = np.random.normal(0, 50, len(actual))
    predictions = actual + noise

    df = pd.DataFrame({
        'Transaction_ID': sample_retail_data['Transaction_ID'],
        'predicted_amount': predictions.round(2)
    })
    return df


@pytest.fixture
def mock_generator_config():
    """Mock configuration for RetailDataGenerator."""
    return {
        'random_seed': 42,
        'num_transactions': 1000,
        'start_date': datetime(2022, 1, 1),
        'end_date': datetime(2023, 12, 31)
    }


@pytest.fixture
def regional_test_data():
    """Create data specifically for regional pattern testing."""
    np.random.seed(42)

    # Create data with known regional spending differences
    # Using larger sample sizes and smaller std dev for statistical significance
    data = []
    regions = {
        'West': {'base_amount': 400, 'count': 500},
        'North': {'base_amount': 350, 'count': 500},
        'East': {'base_amount': 300, 'count': 500},
        'South': {'base_amount': 220, 'count': 500}
    }

    for region, config in regions.items():
        for i in range(config['count']):
            amount = np.random.normal(config['base_amount'], 30)  # Lower std dev
            data.append({
                'Transaction_ID': f'TXN-{len(data):05d}',
                'Customer_ID': f'CUST-{np.random.randint(0, 100):04d}',
                'Gender': np.random.choice(['Male', 'Female']),
                'Age': np.random.randint(18, 75),
                'Category': np.random.choice(['Electronics', 'Clothing', 'Grocery', 'Beauty', 'Furniture']),
                'Quantity': 1,
                'Unit_Price': amount,
                'Discount': 0,
                'Date': datetime(2023, 1, 1),
                'Store_Region': region,
                'Online_Or_Offline': np.random.choice(['Online', 'Offline']),
                'Payment_Method': 'Credit Card',
                'Total_Amount': max(0, amount)
            })

    return pd.DataFrame(data)


@pytest.fixture
def seasonal_test_data():
    """Create data specifically for seasonal pattern testing."""
    np.random.seed(42)

    data = []
    # Q4 should have higher Electronics sales
    for month in range(1, 13):
        quarter = (month - 1) // 3 + 1
        n_transactions = 100

        for i in range(n_transactions):
            category = np.random.choice(
                ['Electronics', 'Clothing', 'Grocery', 'Beauty', 'Furniture'],
                p=[0.2, 0.2, 0.2, 0.2, 0.2]
            )

            # Apply Q4 multiplier to Electronics
            base_amount = 100
            if category == 'Electronics' and quarter == 4:
                base_amount *= 1.35

            data.append({
                'Transaction_ID': f'TXN-{len(data):05d}',
                'Customer_ID': f'CUST-{np.random.randint(0, 100):04d}',
                'Gender': np.random.choice(['Male', 'Female']),
                'Age': np.random.randint(18, 75),
                'Category': category,
                'Quantity': 1,
                'Unit_Price': base_amount + np.random.normal(0, 20),
                'Discount': 0,
                'Date': datetime(2023, month, 15),
                'Store_Region': np.random.choice(['West', 'North', 'East', 'South']),
                'Online_Or_Offline': np.random.choice(['Online', 'Offline']),
                'Payment_Method': 'Credit Card',
                'Total_Amount': base_amount + np.random.normal(0, 20)
            })

    return pd.DataFrame(data)


@pytest.fixture
def customer_id_test_data():
    """Create data for customer ID behavior testing."""
    np.random.seed(42)

    data = []

    # Online customers - more repeat
    online_customers = [f'CUST-{i:04d}' for i in range(50)]
    for _ in range(300):
        # 70% repeat from existing pool
        if np.random.random() < 0.7 and len(online_customers) > 10:
            customer_id = np.random.choice(online_customers[:10])  # Repeat top customers
        else:
            customer_id = np.random.choice(online_customers)

        data.append({
            'Transaction_ID': f'TXN-{len(data):05d}',
            'Customer_ID': customer_id,
            'Gender': 'Female',
            'Age': 35,
            'Category': 'Electronics',
            'Quantity': 1,
            'Unit_Price': 100,
            'Discount': 0,
            'Date': datetime(2023, 1, 1),
            'Store_Region': 'East',
            'Online_Or_Offline': 'Online',
            'Payment_Method': 'Credit Card',
            'Total_Amount': 100
        })

    # Offline customers - more transient
    for _ in range(300):
        # 70% unique
        if np.random.random() < 0.7:
            customer_id = f'CUST-{np.random.randint(1000, 9999):04d}'
        else:
            customer_id = f'CUST-{np.random.randint(100, 200):04d}'  # Loyalty pool

        data.append({
            'Transaction_ID': f'TXN-{len(data):05d}',
            'Customer_ID': customer_id,
            'Gender': 'Male',
            'Age': 45,
            'Category': 'Grocery',
            'Quantity': 1,
            'Unit_Price': 50,
            'Discount': 0,
            'Date': datetime(2023, 1, 1),
            'Store_Region': 'South',
            'Online_Or_Offline': 'Offline',
            'Payment_Method': 'Cash',
            'Total_Amount': 50
        })

    return pd.DataFrame(data)
