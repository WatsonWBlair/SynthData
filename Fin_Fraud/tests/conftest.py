"""Shared fixtures for test suite."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_fraud_metadata():
    """Create sample fraud metadata for testing."""
    return [
        {'index': 10, 'type': 'structuring', 'pattern': 'threshold_avoidance'},
        {'index': 25, 'type': 'wash_trading', 'pattern': 'circular_amounts'},
        {'index': 50, 'type': 'layering', 'pattern': 'round_amount'},
        {'index': 75, 'type': 'market_manipulation', 'pattern': 'large_coordinated'},
        {'index': 90, 'type': 'after_hours', 'pattern': 'suspicious_timing'}
    ]


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    np.random.seed(42)
    n_transactions = 100
    
    data = {
        'Timestamp': pd.date_range('2023-01-01', periods=n_transactions, freq='H'),
        'TransactionID': [f'txn_{i:04d}' for i in range(n_transactions)],
        'AccountID': np.random.randint(1, 16, n_transactions),
        'Amount': np.random.uniform(1000, 100000, n_transactions),
        'Merchant': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_transactions),
        'TransactionType': np.random.choice(['Purchase', 'Transfer', 'Withdrawal'], n_transactions),
        'Location': np.random.choice(['New York', 'London', 'Tokyo'], n_transactions)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_predictions():
    """Create sample prediction data for testing grading tools."""
    predictions = np.zeros(100, dtype=int)
    # Mark some as fraud (indices 8, 12, 25, 51, 88)
    predictions[[8, 12, 25, 51, 88]] = 1
    
    df = pd.DataFrame({
        'TransactionID': [f'txn_{i:04d}' for i in range(100)],
        'is_fraud': predictions
    })
    return df


@pytest.fixture
def mock_generator_config():
    """Mock configuration for DarkPoolDataGenerator."""
    return {
        'random_seed': 42,
        'num_transactions': 1000,
        'fraud_rate': 0.02
    }