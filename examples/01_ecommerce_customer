"""
Example 1: E-commerce Customer Data Cleaning
Real-world example of cleaning customer data from an e-commerce platform
"""

import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append('../src')
from cleaner import DataCleaner
from transformers import DataTransformer
from validators import DataValidator

print("=" * 60)
print("🛒 EXAMPLE 1: E-commerce Customer Data Cleaning")
print("=" * 60)

# Simulate e-commerce customer data
def generate_sample_ecommerce_data():
    """Generate realistic e-commerce customer data with common issues"""
    np.random.seed(42)
    n_rows = 1000
    
    data = {
        'customer_ID': range(1000, 1000 + n_rows),
        'Customer Name': np.random.choice(['John Doe', 'Jane Smith', 'Alex Johnson', None, 'Bob Wilson', 'Sarah Lee'], n_rows),
        'Email': [f'user{i}@example.com' if np.random.random() > 0.1 else np.random.choice(['invalid', None, 'no-email', f'user{i}@']) for i in range(n_rows)],
        'Signup Date': pd.date_range('2020-01-01', periods=n_rows, freq='D').tolist(),
        'Last Purchase Date': pd.date_range('2023-01-01', periods=n_rows, freq='D').tolist(),
        'Total Spend ($)': np.random.exponential(100, n_rows).round(2),
        'Number of Orders': np.random.poisson(5, n_rows),
        'Average Order Value': np.random.uniform(20, 200, n_rows).round(2),
        'Customer Segment': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum', None, 'New'], n_rows),
        'Is Active': np.random.choice([True, False, None], n_rows, p=[0.7, 0.25, 0.05]),
        'Age': np.random.choice(list(range(18, 80)) + [None] * 10, n_rows),
        'Location': np.random.choice(['New York', 'London', 'Tokyo', 'Sydney', 'Berlin', None, 'Paris'], n_rows),
        'Email Opt-in': np.random.choice([True, False, None], n_rows),
        'Loyalty Points': np.random.poisson(500, n_rows),
    }
    
    # Introduce some outliers
    outlier_indices = np.random.choice(n_rows, 20, replace=False)
    for idx in outlier_indices:
        if np.random.random() > 0.5:
            data['Total Spend ($)'][idx] *= 10  # Extreme high spenders
        else:
            data['Age'][idx] = np.random.choice([150, -5, 200])  # Invalid ages
    
    # Introduce duplicates
    duplicate_indices = np.random.choice(n_rows, 50, replace=False)
    for i in range(0, len(duplicate_indices), 2):
        if i + 1 < len(duplicate_indices):
            idx1, idx2 = duplicate_indices[i], duplicate_indices[i+1]
            for col in data:
                if col != 'customer_ID':
                    data[col][idx2] = data[col][idx1]
    
    df = pd.DataFrame(data)
    
    # Save for reference
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_ecommerce.csv', index=False)
    
    return df

def run_ecommerce_example():
    """Run complete e-commerce data cleaning pipeline"""
    
    # Step 1: Load or generate data
    print("\n📥 Step 1: Loading data...")
    if os.path.exists('data/sample_ecommerce.csv'):
        df = pd.read_csv('data/sample_ecommerce.csv')
    else:
        df = generate_sample_ecommerce_data()
    
    print(f"   Original data shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Step 2: Initialize DataCleaner
    print("\n🧹 Step 2: Initializing DataCleaner...")
    cleaner = DataCleaner(df, verbose=True)
    
    # Get initial summary
    print("\n📊 Initial Data Summary:")
    summary = cleaner.get_data_summary()
    print(summary[['column', 'dtype', 'null_percentage', 'unique_values']].to_string())
    
    # Step 3: Standard cleaning pipeline
    print("\n🔧 Step 3: Running cleaning pipeline...")
    
    cleaner.standardize_column_names(case='snake') \
          .handle_missing_values(strategy='auto', threshold=0.2) \
          .convert_data_types() \
          .remove_duplicates(subset=['customer_id', 'email']) \
          .detect_outliers() \
          .handle_outliers(method='cap')
    
    # Step 4: Data transformation
    print("\n🔄 Step 4: Applying transformations...")
    transformer = DataTransformer(verbose=True)
    
    # Extract features from dates
    cleaner.df = transformer.extract_datetime_features(cleaner.df, 'signup_date')
    cleaner.df = transformer.extract_datetime_features(cleaner.df, 'last_purchase_date')
    
    # Calculate customer lifetime
    cleaner.df['customer_lifetime_days'] = (cleaner.df['last_purchase_date'] - cleaner.df['signup_date']).dt.days
    
    # Create derived metrics
    cleaner.df = transformer.create_features(cleaner.df, {
        'orders_per_year': ['number_of_orders / (customer_lifetime_days / 365)'],
        'spend_per_order': ['total_spend / number_of_orders'],
        'is_high_value': ['total_spend > 500']
    })
    
    # Step 5: Data validation
    print("\n🔍 Step 5: Validating data...")
    validator = DataValidator(verbose=True)
    
    # Define business rules
    validation_rules = {
        'schema': {
            'customer_id': int,
            'email': str,
            'age': int,
            'total_spend': float,
            'is_active': bool
        },
        'ranges': {
            'age': {'min': 18, 'max': 100},
            'total_spend': {'min': 0, 'max': 10000},
            'loyalty_points': {'min': 0, 'max': 10000}
        },
        'patterns': {
            'email': r'^[^@]+@[^@]+\.[^@]+$'
        },
        'required': ['customer_id', 'email', 'signup_date']
    }
    
    # Run validations
    schema_results = validator.validate_schema(cleaner.df, validation_rules['schema'])
    range_results = validator.validate_ranges(cleaner.df, validation_rules['ranges'])
    pattern_results = validator.validate_patterns(cleaner.df, validation_rules['patterns'])
    completeness_results = validator.validate_completeness(cleaner.df, validation_rules['required'])
    
    # Step 6: Generate reports
    print("\n📈 Step 6: Generating reports...")
    
    # Data quality score
    from utils import calculate_data_quality_score
    quality_score = calculate_data_quality_score(cleaner.df)
    print(f"\n📊 Data Quality Score: {quality_score}/100")
    
    # Create data dictionary
    from utils import create_data_dictionary
    data_dict = create_data_dictionary(cleaner.df, {
        'customer_id': 'Unique customer identifier',
        'email': 'Customer email address',
        'total_spend': 'Total amount spent by customer',
        'customer_segment': 'Tier based on spending behavior',
        'is_high_value': 'Flag for high-value customers (spend > $500)'
    })
    
    print("\n📋 Data Dictionary (first 5 columns):")
    print(data_dict.head().to_string())
    
    # Step 7: Export results
    print("\n💾 Step 7: Exporting cleaned data...")
    os.makedirs('outputs/cleaned_data', exist_ok=True)
    
    cleaner.export_clean_data('outputs/cleaned_data/ecommerce_cleaned.csv')
    cleaner.export_clean_data('outputs/cleaned_data/ecommerce_cleaned.xlsx')
    
    data_dict.to_csv('outputs/cleaned_data/data_dictionary.csv', index=False)
    
    # Generate cleaning report
    report = cleaner.get_cleaning_report()
    report.to_csv('outputs/cleaned_data/cleaning_report.csv', index=False)
    
    # Generate validation report
    validation_report = validator.generate_validation_report()
    validation_report.to_csv('outputs/cleaned_data/validation_report.csv', index=False)
    
    print("\n✅ E-commerce data cleaning completed!")
    print(f"   Cleaned data shape: {cleaner.df.shape}")
    print(f"   Files saved in 'outputs/cleaned_data/'")
    
    return cleaner.df

# Run the example
if __name__ == "__main__":
    cleaned_data = run_ecommerce_example()
    
    # Show sample of cleaned data
    print("\n🎯 Sample of cleaned data (first 3 rows):")
    print(cleaned_data.head(3).T)
