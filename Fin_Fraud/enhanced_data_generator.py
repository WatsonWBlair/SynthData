"""
Enhanced Financial Dark Pool Data Generator
===========================================

Generates realistic institutional trading data that:
1. Follows natural Benford's Law distribution with slight deviation
2. Models market-maker specific behaviors
3. Injects educational fraud patterns
4. Maintains dark pool transaction characteristics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import random
import os
from typing import Tuple, Dict, List, Optional

# Configuration Constants
DEFAULT_TRANSACTIONS = 216960
DEFAULT_FRAUD_RATE = 0.02
MIN_AMOUNT = 50
MAX_AMOUNT = 1000000
ROUND_LOT_PROBABILITY = 0.3

class DarkPoolDataGenerator:
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Market-maker profiles (Merchants A-J)
        self.market_makers = {
            'A': {'type': 'hft', 'avg_size': 25000, 'frequency': 'high', 'hours': 'market'},
            'B': {'type': 'block', 'avg_size': 150000, 'frequency': 'low', 'hours': 'extended'},
            'C': {'type': 'arbitrage', 'avg_size': 75000, 'frequency': 'medium', 'hours': 'market'},
            'D': {'type': 'institutional', 'avg_size': 200000, 'frequency': 'low', 'hours': 'business'},
            'E': {'type': 'hft', 'avg_size': 30000, 'frequency': 'high', 'hours': 'market'},
            'F': {'type': 'block', 'avg_size': 180000, 'frequency': 'low', 'hours': 'extended'},
            'G': {'type': 'retail_aggregator', 'avg_size': 45000, 'frequency': 'medium', 'hours': 'market'},
            'H': {'type': 'institutional', 'avg_size': 250000, 'frequency': 'low', 'hours': 'business'},
            'I': {'type': 'arbitrage', 'avg_size': 85000, 'frequency': 'medium', 'hours': 'market'},
            'J': {'type': 'specialist', 'avg_size': 120000, 'frequency': 'medium', 'hours': 'extended'}
        }
        
        # Account profiles (institutional investors)
        self.account_profiles = {
            i: {
                'risk_profile': random.choice(['conservative', 'moderate', 'aggressive']),
                'size_preference': random.choice(['small', 'medium', 'large']),
                'activity_level': random.choice(['low', 'medium', 'high'])
            } for i in range(1, 16)
        }
        
        # Transaction locations (major financial centers)
        self.locations = ['New York', 'London', 'Tokyo', 'San Francisco', 'Los Angeles']
        
        # Transaction types in dark pool context
        self.transaction_types = ['Purchase', 'Transfer', 'Withdrawal']
        
    def generate_benford_compliant_amount(self, base_amount: float, market_maker_type: str) -> float:
        """
        Generate amounts that naturally follow Benford's Law with realistic deviations
        """
        # Power law distribution parameters for different market maker types
        type_params = {
            'hft': {'alpha': 2.1, 'scale': 0.8},
            'block': {'alpha': 1.8, 'scale': 1.5},
            'arbitrage': {'alpha': 2.0, 'scale': 1.0},
            'institutional': {'alpha': 1.7, 'scale': 1.8},
            'retail_aggregator': {'alpha': 2.2, 'scale': 0.9},
            'specialist': {'alpha': 1.9, 'scale': 1.2}
        }
        
        params = type_params.get(market_maker_type, {'alpha': 2.0, 'scale': 1.0})
        
        # Generate power-law distributed random number
        u = np.random.uniform(0.1, 1.0)  # Avoid very small numbers
        power_law_factor = (1 - u) ** (-1/params['alpha'])
        
        # Apply to base amount with scaling
        amount = base_amount * power_law_factor * params['scale']
        
        # Add institutional round-lot bias
        if np.random.random() < ROUND_LOT_PROBABILITY:
            if amount > 100000:
                amount = round(amount / 10000) * 10000  # Round to $10K
            elif amount > 10000:
                amount = round(amount / 5000) * 5000   # Round to $5K
            else:
                amount = round(amount / 1000) * 1000   # Round to $1K
        
        # Ensure within realistic bounds
        amount = max(MIN_AMOUNT, min(MAX_AMOUNT, amount))
        
        # Add small random variation to avoid perfect patterns
        variation = np.random.normal(1, 0.05)
        amount *= max(0.8, min(1.2, variation))
        
        # Ensure no negative or zero amounts
        amount = max(MIN_AMOUNT, amount)
        
        return round(amount, 2)
    
    def generate_transaction_timestamp(self, base_date: datetime, market_maker_profile: Dict, is_fraudulent: bool = False) -> datetime:
        """
        Generate realistic timestamps based on market maker behavior
        """
        # Add random days to base date
        days_offset = np.random.randint(0, 365)
        date = base_date + timedelta(days=days_offset)
        
        # Market maker hour preferences
        if market_maker_profile['hours'] == 'market':
            # Market hours 9-17 (8 hours)
            hour = np.random.choice(range(9, 17), p=[0.05, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1])
        elif market_maker_profile['hours'] == 'extended':
            # Extended hours 6-20 (14 hours)
            hour = np.random.choice(range(6, 20), p=[0.03, 0.05, 0.07, 0.1, 0.12, 0.14, 0.14, 0.13, 0.1, 0.07, 0.03, 0.005, 0.005, 0.01])
        else:  # business hours
            # Business hours 8-18 (10 hours)
            hour = np.random.choice(range(8, 18), p=[0.05, 0.1, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.1, 0.03])
        
        # Fraudulent transactions might occur at suspicious times
        if is_fraudulent and np.random.random() < 0.4:
            hour = np.random.choice([2, 3, 4, 22, 23])  # Suspicious hours
        
        minute = np.random.randint(0, 60)
        
        return date.replace(hour=hour, minute=minute, second=0)
    
    def inject_fraud_pattern(self, transactions_df: pd.DataFrame, fraud_rate: float = 0.02) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Inject institutional fraud patterns for educational purposes
        """
        total_transactions = len(transactions_df)
        fraud_count = int(total_transactions * fraud_rate)
        fraud_indices = np.random.choice(total_transactions, fraud_count, replace=False)
        
        fraud_metadata = []
        
        for idx in fraud_indices:
            fraud_type = np.random.choice([
                'structuring', 'wash_trading', 'layering', 
                'market_manipulation', 'after_hours'
            ], p=[0.3, 0.2, 0.2, 0.2, 0.1])
            
            if fraud_type == 'structuring':
                # Just under reporting thresholds
                transactions_df.at[idx, 'Amount'] = np.random.choice([9999.99, 49999.99, 99999.99])
                fraud_metadata.append({'index': idx, 'type': 'structuring', 'pattern': 'threshold_avoidance'})
                
            elif fraud_type == 'wash_trading':
                # Circular trading pattern - same amounts, rapid timing
                # Fix boundary check to prevent index errors
                if idx < total_transactions - 3:
                    base_amount = transactions_df.at[idx, 'Amount']
                    transactions_df.at[idx+1, 'Amount'] = base_amount
                    transactions_df.at[idx+2, 'Amount'] = base_amount
                    # Make timestamps very close
                    base_time = transactions_df.at[idx, 'Timestamp']
                    transactions_df.at[idx+1, 'Timestamp'] = base_time + timedelta(minutes=1)
                    transactions_df.at[idx+2, 'Timestamp'] = base_time + timedelta(minutes=2)
                fraud_metadata.append({'index': idx, 'type': 'wash_trading', 'pattern': 'circular_amounts'})
                
            elif fraud_type == 'layering':
                # Rapid sequence of similar amounts
                transactions_df.at[idx, 'Amount'] = 50000.0  # Suspiciously round
                fraud_metadata.append({'index': idx, 'type': 'layering', 'pattern': 'round_amount'})
                
            elif fraud_type == 'market_manipulation':
                # Coordinated large transactions
                transactions_df.at[idx, 'Amount'] = np.random.uniform(500000, 800000)
                fraud_metadata.append({'index': idx, 'type': 'market_manipulation', 'pattern': 'large_coordinated'})
                
            elif fraud_type == 'after_hours':
                # Suspicious timing
                suspicious_time = transactions_df.at[idx, 'Timestamp'].replace(hour=3, minute=0)
                transactions_df.at[idx, 'Timestamp'] = suspicious_time
                transactions_df.at[idx, 'Amount'] = 100000.0  # Round amount at suspicious time
                fraud_metadata.append({'index': idx, 'type': 'after_hours', 'pattern': 'suspicious_timing'})
        
        return transactions_df, fraud_metadata
    
    def generate_enhanced_dataset(self, num_transactions: int = DEFAULT_TRANSACTIONS, fraud_rate: float = DEFAULT_FRAUD_RATE) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Generate complete enhanced dataset with fraud patterns
        
        Args:
            num_transactions: Number of transactions to generate
            fraud_rate: Percentage of fraudulent transactions (0.0-1.0)
            
        Returns:
            Tuple of (DataFrame with transactions, List of fraud metadata)
        """
        # Validate inputs
        if not 1000 <= num_transactions <= 10000000:
            raise ValueError("num_transactions must be between 1,000 and 10,000,000")
        if not 0 <= fraud_rate <= 0.5:
            raise ValueError("fraud_rate must be between 0 and 0.5")
        """
        Generate complete enhanced dataset
        """
        print(f"Generating {num_transactions:,} enhanced dark pool transactions...")
        
        transactions = []
        fraud_metadata = []
        
        base_date = datetime(2023, 1, 1, 8, 0, 0)
        
        for i in range(num_transactions):
            # Select random market maker and account
            merchant_id = np.random.choice(list(self.market_makers.keys()))
            account_id = np.random.randint(1, 16)
            
            # Get profiles
            market_maker_profile = self.market_makers[merchant_id]
            account_profile = self.account_profiles[account_id]
            
            # Generate base amount based on market maker type
            base_amount = market_maker_profile['avg_size']
            
            # Account size preference adjustment
            if account_profile['size_preference'] == 'small':
                base_amount *= 0.6
            elif account_profile['size_preference'] == 'large':
                base_amount *= 1.8
            
            # Generate Benford-compliant amount
            amount = self.generate_benford_compliant_amount(
                base_amount, market_maker_profile['type']
            )
            
            # Generate realistic timestamp
            timestamp = self.generate_transaction_timestamp(
                base_date, market_maker_profile
            )
            
            # Select transaction details
            transaction_type = np.random.choice(self.transaction_types)
            location = np.random.choice(self.locations)
            
            # Create transaction ID (sequential but realistic)
            transaction_id = 1000 + i + np.random.randint(-50, 50)
            
            transaction = {
                'Timestamp': timestamp,
                'TransactionID': transaction_id,
                'AccountID': account_id,
                'Amount': amount,
                'Merchant': merchant_id,
                'TransactionType': transaction_type,
                'Location': location
            }
            
            transactions.append(transaction)
            
            if (i + 1) % 20000 == 0:
                print(f"Generated {i+1:,} transactions...")
        
        # Create DataFrame
        df = pd.DataFrame(transactions)
        
        # Sort by timestamp for realism
        df = df.sort_values('Timestamp').reset_index(drop=True)
        
        # Inject fraud patterns
        print(f"Injecting fraud patterns ({fraud_rate*100:.1f}% fraud rate)...")
        df, fraud_metadata = self.inject_fraud_pattern(df, fraud_rate)
        
        print("Enhanced dataset generation complete!")
        
        return df, fraud_metadata
    
    def generate_student_dataset(self, num_transactions: int = DEFAULT_TRANSACTIONS, 
                                fraud_rate: float = DEFAULT_FRAUD_RATE,
                                difficulty: str = 'intermediate') -> pd.DataFrame:
        """
        Generate dataset for students WITHOUT fraud labels
        
        Args:
            num_transactions: Number of transactions to generate
            fraud_rate: Percentage of fraudulent transactions
            difficulty: 'beginner' (obvious), 'intermediate', 'advanced' (subtle)
            
        Returns:
            DataFrame with unlabeled transaction data for student analysis
        """
        # Adjust fraud rate based on difficulty
        difficulty_rates = {
            'beginner': min(fraud_rate * 2, 0.1),  # More obvious patterns
            'intermediate': fraud_rate,
            'advanced': fraud_rate * 0.5  # Subtle patterns
        }
        
        actual_fraud_rate = difficulty_rates.get(difficulty, fraud_rate)
        
        # Generate full dataset
        df, fraud_metadata = self.generate_enhanced_dataset(num_transactions, actual_fraud_rate)
        
        # Store metadata separately for teacher reference
        fraud_df = pd.DataFrame(fraud_metadata)
        teacher_file = f'./data/teacher_answer_key_{difficulty}.csv'
        fraud_df.to_csv(teacher_file, index=False)
        print(f"Teacher answer key saved to: {teacher_file} (DO NOT SHARE WITH STUDENTS)")
        
        # Return unlabeled data for students
        return df

def generate_enhanced_data():
    """
    Main function to generate enhanced dataset
    """
    generator = DarkPoolDataGenerator(random_seed=42)
    
    # Generate enhanced dataset
    enhanced_df, fraud_metadata = generator.generate_enhanced_dataset(
        num_transactions=216960, 
        fraud_rate=0.02  # 2% fraud rate for educational purposes
    )
    
    # Use relative paths for portability
    data_dir = './data'
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Save enhanced raw data
    output_path = os.path.join(data_dir, 'enhanced_raw_data.csv')
    enhanced_df.to_csv(output_path, index=False)
    print(f"Enhanced dataset saved to: {output_path}")
    
    # Save fraud metadata for validation
    fraud_df = pd.DataFrame(fraud_metadata)
    fraud_path = os.path.join(data_dir, 'fraud_patterns_metadata.csv')
    fraud_df.to_csv(fraud_path, index=False)
    print(f"Fraud metadata saved to: {fraud_path}")
    
    return enhanced_df, fraud_metadata

if __name__ == "__main__":
    enhanced_data, fraud_info = generate_enhanced_data()