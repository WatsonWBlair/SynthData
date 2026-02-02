#!/usr/bin/env python3
"""
Retail Data Generator with Realistic Patterns
==============================================

Generates synthetic retail transaction data with:
1. Regional differentiation (income, urban ratio, online preferences)
2. Seasonal patterns (quarterly, holidays, pay cycles, day-of-week)
3. Customer ID behavior (persistent online vs transient in-store)
4. Demographic patterns (age, gender preferences, basket sizes)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import uuid
import argparse
from typing import Tuple, Dict, List, Optional

# Configuration Constants
DEFAULT_TRANSACTIONS = 50000
DEFAULT_START_DATE = datetime(2020, 1, 1)
DEFAULT_END_DATE = datetime(2023, 12, 31)


class RetailDataGenerator:
    """Generator for realistic retail transaction data with statistical patterns."""

    def __init__(self, random_seed: int = 42):
        """
        Initialize the generator with profile dictionaries.

        Args:
            random_seed: Seed for reproducibility
        """
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.random_seed = random_seed

        # Reset customer registries for deterministic behavior
        self._online_customer_registry = {}
        self._loyalty_customer_registry = {}

        # Regional profiles with economic and behavioral characteristics
        self.regional_profiles = {
            'West': {
                'income_multiplier': 1.25,
                'urban_ratio': 0.75,
                'online_preference': 0.55,
                'category_bias': {'Electronics': 1.3, 'Beauty': 1.2, 'Clothing': 1.0, 'Grocery': 0.9, 'Furniture': 1.1},
                'population_weight': 0.25
            },
            'North': {
                'income_multiplier': 1.15,
                'urban_ratio': 0.70,
                'online_preference': 0.50,
                'category_bias': {'Electronics': 1.0, 'Beauty': 1.0, 'Clothing': 1.25, 'Grocery': 1.0, 'Furniture': 1.0},
                'population_weight': 0.25
            },
            'East': {
                'income_multiplier': 1.0,
                'urban_ratio': 0.60,
                'online_preference': 0.45,
                'category_bias': {'Electronics': 1.0, 'Beauty': 1.0, 'Clothing': 1.0, 'Grocery': 1.0, 'Furniture': 1.0},
                'population_weight': 0.30
            },
            'South': {
                'income_multiplier': 0.85,
                'urban_ratio': 0.45,
                'online_preference': 0.35,
                'category_bias': {'Electronics': 0.9, 'Beauty': 0.95, 'Clothing': 0.85, 'Grocery': 1.2, 'Furniture': 0.95},
                'population_weight': 0.20
            }
        }

        # Category profiles with pricing and seasonal characteristics
        self.category_profiles = {
            'Electronics': {
                'base_price_range': (50, 800),
                'avg_quantity_range': (1, 3),
                'seasonal_multipliers': {'Q1': 0.85, 'Q2': 0.90, 'Q3': 0.95, 'Q4': 1.35},
                'weekend_multiplier': 1.15,
                'discount_range': (0.0, 0.25),
                'age_preference': {'18-25': 1.3, '26-35': 1.25, '36-45': 1.1, '46-55': 0.95, '56-65': 0.85, '65+': 0.7},
                'gender_preference': {'Male': 1.2, 'Female': 0.85}
            },
            'Clothing': {
                'base_price_range': (20, 400),
                'avg_quantity_range': (1, 5),
                'seasonal_multipliers': {'Q1': 1.05, 'Q2': 1.15, 'Q3': 1.20, 'Q4': 1.10},
                'weekend_multiplier': 1.25,
                'discount_range': (0.05, 0.40),
                'age_preference': {'18-25': 1.35, '26-35': 1.25, '36-45': 1.0, '46-55': 0.90, '56-65': 0.80, '65+': 0.70},
                'gender_preference': {'Male': 0.85, 'Female': 1.25}
            },
            'Grocery': {
                'base_price_range': (5, 100),
                'avg_quantity_range': (3, 15),
                'seasonal_multipliers': {'Q1': 1.0, 'Q2': 1.0, 'Q3': 1.05, 'Q4': 1.15},
                'weekend_multiplier': 1.30,
                'discount_range': (0.0, 0.15),
                'age_preference': {'18-25': 0.8, '26-35': 1.0, '36-45': 1.15, '46-55': 1.2, '56-65': 1.15, '65+': 1.1},
                'gender_preference': {'Male': 0.95, 'Female': 1.05}
            },
            'Beauty': {
                'base_price_range': (10, 200),
                'avg_quantity_range': (1, 4),
                'seasonal_multipliers': {'Q1': 0.95, 'Q2': 1.10, 'Q3': 1.05, 'Q4': 1.20},
                'weekend_multiplier': 1.10,
                'discount_range': (0.05, 0.30),
                'age_preference': {'18-25': 1.25, '26-35': 1.3, '36-45': 1.15, '46-55': 1.0, '56-65': 0.85, '65+': 0.75},
                'gender_preference': {'Male': 0.65, 'Female': 1.50}
            },
            'Furniture': {
                'base_price_range': (100, 1500),
                'avg_quantity_range': (1, 3),
                'seasonal_multipliers': {'Q1': 0.90, 'Q2': 1.05, 'Q3': 1.00, 'Q4': 1.10},
                'weekend_multiplier': 1.20,
                'discount_range': (0.05, 0.35),
                'age_preference': {'18-25': 0.6, '26-35': 1.1, '36-45': 1.35, '46-55': 1.2, '56-65': 1.0, '65+': 0.85},
                'gender_preference': {'Male': 1.0, 'Female': 1.0}
            }
        }

        # Age group profiles with spending characteristics
        self.age_group_profiles = {
            '18-25': {'spending_multiplier': 0.80, 'online_preference': 0.70, 'digital_wallet_preference': 0.65},
            '26-35': {'spending_multiplier': 1.10, 'online_preference': 0.65, 'digital_wallet_preference': 0.55},
            '36-45': {'spending_multiplier': 1.25, 'online_preference': 0.55, 'digital_wallet_preference': 0.45},
            '46-55': {'spending_multiplier': 1.15, 'online_preference': 0.45, 'digital_wallet_preference': 0.35},
            '56-65': {'spending_multiplier': 1.00, 'online_preference': 0.35, 'digital_wallet_preference': 0.25},
            '65+': {'spending_multiplier': 0.85, 'online_preference': 0.25, 'digital_wallet_preference': 0.15}
        }

        # Holiday calendar with impact multipliers
        self.seasonal_calendar = {
            # Black Friday week (last week of November)
            'black_friday': {
                'start_offset': lambda year: self._get_black_friday(year) - timedelta(days=2),
                'end_offset': lambda year: self._get_black_friday(year) + timedelta(days=2),
                'multiplier': 2.25,
                'categories': ['Electronics', 'Clothing', 'Furniture', 'Beauty']
            },
            # Christmas season (Dec 15-24)
            'christmas': {
                'start_month_day': (12, 15),
                'end_month_day': (12, 24),
                'multiplier': 1.65,
                'categories': ['Electronics', 'Clothing', 'Beauty', 'Furniture']
            },
            # Back to school (Aug 1-31)
            'back_to_school': {
                'start_month_day': (8, 1),
                'end_month_day': (8, 31),
                'multiplier': 1.55,
                'categories': ['Electronics', 'Clothing']
            },
            # Valentine's Day (Feb 10-14)
            'valentines': {
                'start_month_day': (2, 10),
                'end_month_day': (2, 14),
                'multiplier': 1.40,
                'categories': ['Beauty', 'Clothing']
            },
            # Summer sales (Jun 15 - Jul 15)
            'summer_sales': {
                'start_month_day': (6, 15),
                'end_month_day': (7, 15),
                'multiplier': 1.25,
                'categories': ['Clothing', 'Furniture']
            }
        }

        # Payment methods available
        self.payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet']

    def _get_black_friday(self, year: int) -> datetime:
        """Calculate Black Friday date for a given year (4th Thursday of November)."""
        # Find first day of November
        nov_first = datetime(year, 11, 1)
        # Find first Thursday
        days_until_thursday = (3 - nov_first.weekday()) % 7
        first_thursday = nov_first + timedelta(days=days_until_thursday)
        # Fourth Thursday
        black_friday = first_thursday + timedelta(weeks=3, days=1)  # Day after Thanksgiving
        return black_friday

    def _get_age_group(self, age: int) -> str:
        """Map age to age group."""
        if age < 26:
            return '18-25'
        elif age < 36:
            return '26-35'
        elif age < 46:
            return '36-45'
        elif age < 56:
            return '46-55'
        elif age < 66:
            return '56-65'
        else:
            return '65+'

    def _get_quarter(self, date: datetime) -> str:
        """Get quarter label for a date."""
        month = date.month
        if month <= 3:
            return 'Q1'
        elif month <= 6:
            return 'Q2'
        elif month <= 9:
            return 'Q3'
        else:
            return 'Q4'

    def _get_day_of_month_multiplier(self, day: int) -> float:
        """
        Get pay cycle multiplier based on day of month.
        Month start: 1.25x, mid-month: 0.9x, month end: 1.15x
        """
        if day <= 5:
            return 1.25
        elif day <= 10:
            return 1.15
        elif day <= 20:
            return 0.90
        elif day <= 25:
            return 1.05
        else:
            return 1.15

    def _is_holiday_period(self, date: datetime) -> Tuple[bool, float, List[str]]:
        """
        Check if date falls in a holiday period.
        Returns: (is_holiday, multiplier, affected_categories)
        """
        year = date.year

        for holiday_name, holiday_info in self.seasonal_calendar.items():
            if 'start_offset' in holiday_info:
                # Dynamic date calculation (e.g., Black Friday)
                start = holiday_info['start_offset'](year)
                end = holiday_info['end_offset'](year)
            else:
                # Fixed month/day
                start = datetime(year, holiday_info['start_month_day'][0],
                               holiday_info['start_month_day'][1])
                end = datetime(year, holiday_info['end_month_day'][0],
                             holiday_info['end_month_day'][1])

            if start <= date <= end:
                return True, holiday_info['multiplier'], holiday_info['categories']

        return False, 1.0, []

    def generate_customer_id(self, is_online: bool, region: str) -> str:
        """
        Generate customer ID with persistent vs transient behavior.

        Online: ~70% repeat customers from registry
        In-Store: ~70% new IDs per visit, ~30% loyalty card holders

        Args:
            is_online: Whether the transaction is online
            region: Store region

        Returns:
            Customer ID string
        """
        if is_online:
            # Online customers: 70% repeat, 30% new
            if np.random.random() < 0.70 and len(self._online_customer_registry.get(region, [])) > 0:
                # Return existing customer
                return np.random.choice(self._online_customer_registry[region])
            else:
                # Create new customer
                customer_id = f"CUST-{np.random.randint(0, 9999):04d}"
                if region not in self._online_customer_registry:
                    self._online_customer_registry[region] = []
                # Keep registry to reasonable size (max 500 per region)
                if len(self._online_customer_registry[region]) < 500:
                    self._online_customer_registry[region].append(customer_id)
                return customer_id
        else:
            # In-store: 30% loyalty card holders, 70% transient
            if np.random.random() < 0.30:
                # Loyalty customer
                if len(self._loyalty_customer_registry.get(region, [])) > 0 and np.random.random() < 0.80:
                    return np.random.choice(self._loyalty_customer_registry[region])
                else:
                    customer_id = f"CUST-{np.random.randint(0, 9999):04d}"
                    if region not in self._loyalty_customer_registry:
                        self._loyalty_customer_registry[region] = []
                    if len(self._loyalty_customer_registry[region]) < 200:
                        self._loyalty_customer_registry[region].append(customer_id)
                    return customer_id
            else:
                # Transient customer - always new ID
                return f"CUST-{np.random.randint(0, 9999):04d}"

    def generate_transaction_date(self, start_date: datetime, end_date: datetime,
                                  category: str, region: str) -> datetime:
        """
        Generate seasonality-aware transaction date.

        Considers:
        - Quarterly patterns by category
        - Holiday periods
        - Day of week (weekend boost)
        - Pay cycle patterns

        Args:
            start_date: Start of date range
            end_date: End of date range
            category: Product category
            region: Store region

        Returns:
            Transaction datetime
        """
        # Generate base date with weighted sampling based on patterns
        total_days = (end_date - start_date).days

        # Build probability weights for each day
        date_weights = []
        for day_offset in range(total_days + 1):
            current_date = start_date + timedelta(days=day_offset)
            weight = 1.0

            # Apply quarterly multiplier
            quarter = self._get_quarter(current_date)
            if category in self.category_profiles:
                weight *= self.category_profiles[category]['seasonal_multipliers'].get(quarter, 1.0)

            # Apply holiday multiplier
            is_holiday, holiday_mult, affected_cats = self._is_holiday_period(current_date)
            if is_holiday and category in affected_cats:
                weight *= holiday_mult

            # Apply day of week multiplier (weekend boost)
            if current_date.weekday() >= 5:  # Saturday or Sunday
                if category in self.category_profiles:
                    weight *= self.category_profiles[category]['weekend_multiplier']

            # Apply pay cycle multiplier
            weight *= self._get_day_of_month_multiplier(current_date.day)

            date_weights.append(max(0.1, weight))  # Ensure non-zero weights

        # Normalize weights
        date_weights = np.array(date_weights)
        date_weights = date_weights / date_weights.sum()

        # Sample a day
        selected_day = np.random.choice(total_days + 1, p=date_weights)
        return start_date + timedelta(days=int(selected_day))

    def generate_unit_price(self, category: str, region: str) -> float:
        """
        Generate cost-of-living adjusted unit price.

        Args:
            category: Product category
            region: Store region

        Returns:
            Unit price as float
        """
        cat_profile = self.category_profiles.get(category, self.category_profiles['Grocery'])
        region_profile = self.regional_profiles.get(region, self.regional_profiles['East'])

        # Get base price range
        min_price, max_price = cat_profile['base_price_range']

        # Apply log-normal distribution for more realistic pricing
        mean_log = np.log((min_price + max_price) / 2)
        std_log = 0.5
        base_price = np.random.lognormal(mean_log, std_log)

        # Clip to range
        base_price = np.clip(base_price, min_price, max_price)

        # Apply regional income multiplier (higher income = slightly higher prices)
        regional_adjustment = 0.9 + (region_profile['income_multiplier'] - 0.85) * 0.25
        adjusted_price = base_price * regional_adjustment

        return round(adjusted_price, 2)

    def generate_discount(self, category: str, region: str, date: datetime) -> float:
        """
        Generate seasonal discount pattern.

        Args:
            category: Product category
            region: Store region
            date: Transaction date

        Returns:
            Discount as decimal (0.0 to 0.5)
        """
        cat_profile = self.category_profiles.get(category, self.category_profiles['Grocery'])
        min_discount, max_discount = cat_profile['discount_range']

        # Base discount from beta distribution (skewed towards lower discounts)
        base_discount = np.random.beta(2, 5) * (max_discount - min_discount) + min_discount

        # Holiday boost
        is_holiday, holiday_mult, affected_cats = self._is_holiday_period(date)
        if is_holiday and category in affected_cats:
            # Increase discount probability during holidays
            if np.random.random() < 0.6:
                base_discount *= 1.5

        # Cap discount
        return min(round(base_discount, 2), 0.50)

    def generate_quantity(self, category: str, age_group: str) -> int:
        """
        Generate quantity based on category and demographic.

        Args:
            category: Product category
            age_group: Customer age group

        Returns:
            Quantity as integer
        """
        cat_profile = self.category_profiles.get(category, self.category_profiles['Grocery'])
        min_qty, max_qty = cat_profile['avg_quantity_range']

        # Use Poisson-like distribution centered on the range
        mean_qty = (min_qty + max_qty) / 2
        quantity = np.random.poisson(mean_qty)

        # Clip to range
        return max(1, min(quantity, max_qty + 2))

    def select_category(self, region: str, age: int, gender: str) -> str:
        """
        Select category based on regional and demographic preferences.

        Args:
            region: Store region
            age: Customer age
            gender: Customer gender

        Returns:
            Product category string
        """
        region_profile = self.regional_profiles.get(region, self.regional_profiles['East'])
        age_group = self._get_age_group(age)

        categories = list(self.category_profiles.keys())
        weights = []

        for cat in categories:
            weight = 1.0

            # Apply regional category bias
            weight *= region_profile['category_bias'].get(cat, 1.0)

            # Apply age preference
            cat_profile = self.category_profiles[cat]
            weight *= cat_profile['age_preference'].get(age_group, 1.0)

            # Apply gender preference
            weight *= cat_profile['gender_preference'].get(gender, 1.0)

            weights.append(weight)

        # Normalize and select
        weights = np.array(weights)
        weights = weights / weights.sum()

        return np.random.choice(categories, p=weights)

    def select_payment_method(self, is_online: bool, age_group: str) -> str:
        """
        Select payment method based on channel and demographics.

        Args:
            is_online: Whether transaction is online
            age_group: Customer age group

        Returns:
            Payment method string
        """
        age_profile = self.age_group_profiles.get(age_group, self.age_group_profiles['36-45'])
        digital_pref = age_profile['digital_wallet_preference']

        if is_online:
            # Online: No cash option
            methods = ['Credit Card', 'Debit Card', 'Digital Wallet']
            weights = [0.35, 0.30, 0.35]
            # Adjust for digital preference
            weights[2] *= (1 + digital_pref)
        else:
            # In-store: All options
            methods = ['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet']
            weights = [0.30, 0.25, 0.30, 0.15]
            # Adjust for digital preference
            weights[3] *= (1 + digital_pref)
            # Older customers prefer cash more
            if age_group in ['56-65', '65+']:
                weights[2] *= 1.3

        weights = np.array(weights)
        weights = weights / weights.sum()

        return np.random.choice(methods, p=weights)

    def generate_retail_dataset(self, num_transactions: int = DEFAULT_TRANSACTIONS,
                                start_date: datetime = DEFAULT_START_DATE,
                                end_date: datetime = DEFAULT_END_DATE) -> pd.DataFrame:
        """
        Generate complete retail dataset with realistic patterns.

        Args:
            num_transactions: Number of transactions to generate
            start_date: Start date for transactions
            end_date: End date for transactions

        Returns:
            DataFrame with retail transactions
        """
        if not 100 <= num_transactions <= 10000000:
            raise ValueError("num_transactions must be between 100 and 10,000,000")

        # Reset random state and customer registries for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        self._online_customer_registry = {}
        self._loyalty_customer_registry = {}

        print(f"Generating {num_transactions:,} retail transactions...")

        transactions = []

        # Pre-compute region weights
        regions = list(self.regional_profiles.keys())
        region_weights = [self.regional_profiles[r]['population_weight'] for r in regions]
        region_weights = np.array(region_weights) / sum(region_weights)

        for i in range(num_transactions):
            # Select region based on population weight
            region = np.random.choice(regions, p=region_weights)
            region_profile = self.regional_profiles[region]

            # Generate demographic attributes
            gender = np.random.choice(['Male', 'Female'], p=[0.48, 0.52])
            age = int(np.clip(np.random.normal(42, 15), 18, 85))
            age_group = self._get_age_group(age)
            age_profile = self.age_group_profiles[age_group]

            # Determine online vs offline based on region and age
            base_online_prob = region_profile['online_preference']
            age_online_adjustment = age_profile['online_preference']
            online_prob = (base_online_prob + age_online_adjustment) / 2
            is_online = np.random.random() < online_prob
            channel = 'Online' if is_online else 'Offline'

            # Generate customer ID
            customer_id = self.generate_customer_id(is_online, region)

            # Select category based on demographics
            category = self.select_category(region, age, gender)

            # Generate transaction date
            transaction_date = self.generate_transaction_date(start_date, end_date, category, region)

            # Generate pricing
            unit_price = self.generate_unit_price(category, region)
            discount = self.generate_discount(category, region, transaction_date)
            quantity = self.generate_quantity(category, age_group)

            # Apply demographic spending multiplier
            spending_mult = age_profile['spending_multiplier']
            unit_price *= spending_mult
            unit_price = round(unit_price, 2)

            # Calculate total
            total_amount = round(unit_price * quantity * (1 - discount), 2)

            # Select payment method
            payment_method = self.select_payment_method(is_online, age_group)

            # Create transaction record
            transaction = {
                'Transaction_ID': f'TXN-{i:05d}',
                'Customer_ID': customer_id,
                'Gender': gender,
                'Age': age,
                'Category': category,
                'Quantity': quantity,
                'Unit_Price': unit_price,
                'Discount': discount,
                'Date': transaction_date,
                'Store_Region': region,
                'Online_Or_Offline': channel,
                'Payment_Method': payment_method,
                'Total_Amount': total_amount
            }

            transactions.append(transaction)

            if (i + 1) % 10000 == 0:
                print(f"Generated {i+1:,} transactions...")

        # Create DataFrame
        df = pd.DataFrame(transactions)

        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)

        # Re-assign sequential transaction IDs after sorting
        df['Transaction_ID'] = [f'TXN-{i:05d}' for i in range(len(df))]

        print(f"Dataset generation complete: {len(df):,} transactions")

        return df

    def generate_student_datasets(self, num_datasets: int = 1,
                                  num_transactions: int = DEFAULT_TRANSACTIONS,
                                  start_date: datetime = DEFAULT_START_DATE,
                                  end_date: datetime = DEFAULT_END_DATE,
                                  base_seed: int = 42) -> Dict[str, List[str]]:
        """
        Generate multiple unique datasets for different student teams.

        Args:
            num_datasets: Number of unique datasets to generate
            num_transactions: Number of transactions per dataset
            start_date: Start date for transactions
            end_date: End date for transactions
            base_seed: Base random seed (each dataset uses base_seed + index)

        Returns:
            Dictionary with lists of generated data files
        """
        data_dir = './data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        student_dir = os.path.join(data_dir, 'student_datasets')
        if not os.path.exists(student_dir):
            os.makedirs(student_dir)

        generated_files = {
            'data_files': [],
            'mapping': []
        }

        print(f"\nGenerating {num_datasets} unique student datasets...")

        for i in range(num_datasets):
            seed = base_seed + i
            team_id = f"{i+1:03d}"

            print(f"\nGenerating dataset for Team {team_id} (seed={seed})...")

            # Create new generator with unique seed
            generator = RetailDataGenerator(random_seed=seed)

            # Generate unique dataset
            df = generator.generate_retail_dataset(num_transactions, start_date, end_date)

            # Save dataset
            data_file = os.path.join(student_dir, f'team_{team_id}_data.csv')
            df.to_csv(data_file, index=False)
            generated_files['data_files'].append(data_file)

            # Add to mapping
            generated_files['mapping'].append({
                'team_id': team_id,
                'data_file': data_file,
                'seed': seed,
                'transactions': num_transactions,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            })

        # Save mapping file
        mapping_df = pd.DataFrame(generated_files['mapping'])
        mapping_file = os.path.join(student_dir, 'dataset_mapping.csv')
        mapping_df.to_csv(mapping_file, index=False)

        print(f"\nGenerated {num_datasets} unique datasets")
        print(f"Data files in: {student_dir}/team_XXX_data.csv")
        print(f"Mapping file: {mapping_file}")

        return generated_files


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic retail transaction datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate default dataset (50k transactions)
  python retail_data_generator.py

  # Generate with custom parameters
  python retail_data_generator.py --transactions 100000 --seed 123

  # Generate multiple student datasets
  python retail_data_generator.py --student-datasets 10 --transactions 25000

  # Specify date range
  python retail_data_generator.py --start-date 2022-01-01 --end-date 2023-12-31
        ''')

    parser.add_argument('--transactions', type=int, default=DEFAULT_TRANSACTIONS,
                       help=f'Number of transactions to generate (default: {DEFAULT_TRANSACTIONS})')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--student-datasets', type=int, default=0,
                       help='Generate N unique datasets for students (default: 0 = single dataset)')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date for transactions (YYYY-MM-DD, default: 2020-01-01)')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                       help='End date for transactions (YYYY-MM-DD, default: 2023-12-31)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: ./data/retail_data.csv)')

    args = parser.parse_args()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        parser.error(f'Invalid date format: {e}')

    # Validate arguments
    if not 100 <= args.transactions <= 10000000:
        parser.error('Transactions must be between 100 and 10,000,000')
    if start_date >= end_date:
        parser.error('Start date must be before end date')
    if args.student_datasets < 0:
        parser.error('Number of student datasets must be non-negative')

    if args.student_datasets > 0:
        # Generate multiple student datasets
        generator = RetailDataGenerator(random_seed=args.seed)
        generator.generate_student_datasets(
            num_datasets=args.student_datasets,
            num_transactions=args.transactions,
            start_date=start_date,
            end_date=end_date,
            base_seed=args.seed
        )
    else:
        # Generate single dataset
        generator = RetailDataGenerator(random_seed=args.seed)

        print(f"Generating {args.transactions:,} transactions...")
        df = generator.generate_retail_dataset(
            num_transactions=args.transactions,
            start_date=start_date,
            end_date=end_date
        )

        # Save data
        data_dir = './data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        output_path = args.output or os.path.join(data_dir, 'retail_data.csv')
        df.to_csv(output_path, index=False)
        print(f"\nDataset saved to: {output_path}")

        # Print summary statistics
        print("\nDataset Summary:")
        print(f"  Total transactions: {len(df):,}")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"  Regions: {df['Store_Region'].value_counts().to_dict()}")
        print(f"  Categories: {df['Category'].value_counts().to_dict()}")
        print(f"  Online vs Offline: {df['Online_Or_Offline'].value_counts().to_dict()}")
        print(f"  Average Total: ${df['Total_Amount'].mean():.2f}")

        return df


if __name__ == "__main__":
    main()
