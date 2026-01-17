"""
Example 3: Customer Churn Prediction Data Preparation
Preparing data for machine learning churn prediction models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append('../src')
from cleaner import DataCleaner
from transformers import DataTransformer
from validators import DataValidator

print("=" * 60)
print("📉 EXAMPLE 3: Customer Churn Prediction Data Prep")
print("=" * 60)

def generate_churn_data():
    """Generate customer churn dataset with realistic patterns"""
    np.random.seed(42)
    n_customers = 2000
    
    # Generate customer IDs
    customer_ids = [f'CUST{10000 + i}' for i in range(n_customers)]
    
    # Demographic data
    ages = np.random.normal(40, 15, n_customers).astype(int)
    ages = np.clip(ages, 18, 80)
    
    genders = np.random.choice(['Male', 'Female', 'Other'], n_customers, p=[0.48, 0.48, 0.04])
    
    # Signup dates (over last 3 years)
    start_date = datetime(2020, 1, 1)
    signup_dates = [
        start_date + timedelta(days=np.random.randint(0, 1095))
        for _ in range(n_customers)
    ]
    
    # Subscription data
    subscription_types = np.random.choice(['Basic', 'Premium', 'Enterprise', 'Trial'], 
                                        n_customers, p=[0.5, 0.3, 0.15, 0.05])
    
    # Payment data with some missing
    monthly_payments = []
    for sub_type in subscription_types:
        if sub_type == 'Basic':
            monthly_payments.append(np.random.uniform(9.99, 19.99))
        elif sub_type == 'Premium':
            monthly_payments.append(np.random.uniform(29.99, 49.99))
        elif sub_type == 'Enterprise':
            monthly_payments.append(np.random.uniform(99.99, 199.99))
        else:  # Trial
            monthly_payments.append(0)
    
    # Usage metrics (with patterns that predict churn)
    avg_session_duration = np.random.exponential(30, n_customers)  # minutes
    sessions_per_week = np.random.poisson(5, n_customers)
    
    # Support tickets (churn indicator)
    support_tickets = np.random.poisson(2, n_customers)
    
    # Payment failures (churn indicator)
    payment_failures = np.random.poisson(0.5, n_customers)
    
    # Customer satisfaction (1-5)
    satisfaction_scores = np.random.choice([1, 2, 3, 4, 5, None], n_customers, 
                                          p=[0.05, 0.1, 0.15, 0.3, 0.35, 0.05])
    
    # Last login date (some recent, some old)
    days_since_last_login = np.random.exponential(30, n_customers).astype(int)
    last_login_dates = [
        datetime(2023, 12, 31) - timedelta(days=int(days))
        for days in days_since_last_login
    ]
    
    # Churn label (target variable)
    # Create realistic churn patterns based on features
    churn_probabilities = []
    for i in range(n_customers):
        prob = 0.1  # Base churn rate
        
        # Factors increasing churn probability
        if subscription_types[i] == 'Trial':
            prob += 0.3
        if support_tickets[i] > 3:
            prob += 0.2
        if payment_failures[i] > 0:
            prob += 0.15
        if satisfaction_scores[i] is not None and satisfaction_scores[i] < 3:
            prob += 0.25
        if days_since_last_login[i] > 60:
            prob += 0.3
        
        # Factors decreasing churn probability
        if subscription_types[i] == 'Enterprise':
            prob -= 0.1
        if sessions_per_week[i] > 7:
            prob -= 0.15
        
        churn_probabilities.append(min(max(prob, 0), 0.9))
    
    # Generate churn labels based on probabilities
    churned = [np.random.random() < prob for prob in churn_probabilities]
    
    # Create dataset
    data = {
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'signup_date': signup_dates,
        'subscription_type': subscription_types,
        'monthly_payment': monthly_payments,
        'avg_session_duration': avg_session_duration.round(1),
        'sessions_per_week': sessions_per_week,
        'support_tickets_last_month': support_tickets,
        'payment_failures_last_quarter': payment_failures,
        'satisfaction_score': satisfaction_scores,
        'last_login_date': last_login_dates,
        'churned': churned
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    missing_mask = np.random.random(size=df.shape) < 0.05
    df = df.mask(missing_mask)
    
    # Save data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_churn.csv', index=False)
    
    return df

def run_churn_analysis():
    """Prepare data for churn prediction modeling"""
    
    print("\n📥 Step 1: Loading churn prediction data...")
    if os.path.exists('data/sample_churn.csv'):
        df = pd.read_csv('data/sample_churn.csv', 
                        parse_dates=['signup_date', 'last_login_date'])
    else:
        df = generate_churn_data()
    
    print(f"   Customers: {df.shape[0]}, Features: {df.shape[1]}")
    print(f"   Churn rate: {df['churned'].mean():.1%}")
    
    # Initialize components
    cleaner = DataCleaner(df, verbose=True)
    transformer = DataTransformer(verbose=True)
    validator = DataValidator(verbose=True)
    
    print("\n🔧 Step 2: Data cleaning...")
    
    cleaner.standardize_column_names(case='snake') \
          .handle_missing_values(strategy='auto', threshold=0.1) \
          .convert_data_types() \
          .remove_duplicates(subset=['customer_id'])
    
    # Fix data quality issues
    print("\n🎯 Step 3: Fixing data quality issues...")
    
    # Ensure ages are realistic
    invalid_age_mask = (cleaner.df['age'] < 18) | (cleaner.df['age'] > 100)
    if invalid_age_mask.any():
        print(f"   Found {invalid_age_mask.sum()} invalid ages")
        cleaner.df.loc[invalid_age_mask, 'age'] = cleaner.df['age'].median()
    
    # Fix session duration outliers
    cleaner.detect_outliers(method='iqr', threshold=3)
    cleaner.handle_outliers(method='cap')
    
    print("\n🔄 Step 4: Feature engineering for churn prediction...")
    
    # Extract features from dates
    cleaner.df = transformer.extract_datetime_features(cleaner.df, 'signup_date')
    cleaner.df = transformer.extract_datetime_features(cleaner.df, 'last_login_date')
    
    # Calculate tenure in days
    reference_date = datetime(2023, 12, 31)
    cleaner.df['tenure_days'] = (reference_date - cleaner.df['signup_date']).dt.days
    cleaner.df['days_since_last_login'] = (reference_date - cleaner.df['last_login_date']).dt.days
    
    # Create engagement score
    cleaner.df = transformer.create_features(cleaner.df, {
        'engagement_score': [
            'avg_session_duration * sessions_per_week / 100'
        ],
        'is_active_user': [
            'days_since_last_login < 30'
        ],
        'customer_value': [
            'tenure_days * monthly_payment / 365'
        ],
        'risk_score': [
            'support_tickets_last_month * 0.3 + payment_failures_last_quarter * 0.5'
        ]
    })
    
    # Handle categorical variables
    cleaner.df = transformer.encode_categorical(
        cleaner.df, 
        columns=['gender', 'subscription_type'], 
        method='onehot'
    )
    
    # Normalize numeric features for ML
    numeric_cols = [
        'age', 'monthly_payment', 'avg_session_duration',
        'sessions_per_week', 'tenure_days', 'days_since_last_login',
        'engagement_score', 'customer_value', 'risk_score'
    ]
    
    cleaner.df = transformer.normalize_numeric(
        cleaner.df, 
        columns=numeric_cols, 
        method='standard'
    )
    
    print("\n🔍 Step 5: Data validation for ML...")
    
    # Validate no missing values in features
    missing_report = cleaner.df.isna().sum()
    missing_cols = missing_report[missing_report > 0]
    
    if len(missing_cols) > 0:
        print(f"   ⚠️  Found missing values in: {list(missing_cols.index)}")
        # Fill remaining missing values
        for col in missing_cols.index:
            if pd.api.types.is_numeric_dtype(cleaner.df[col]):
                cleaner.df[col].fillna(cleaner.df[col].median(), inplace=True)
            else:
                cleaner.df[col].fillna(cleaner.df[col].mode()[0], inplace=True)
    
    # Check class balance
    churn_rate = cleaner.df['churned'].mean()
    print(f"   Churn rate after cleaning: {churn_rate:.1%}")
    
    if churn_rate < 0.2 or churn_rate > 0.8:
        print("   ⚠️  Warning: Highly imbalanced dataset")
    
    print("\n📊 Step 6: Feature analysis...")
    
    # Calculate correlation with churn
    numeric_features = cleaner.df.select_dtypes(include=[np.number]).columns
    correlation_with_churn = cleaner.df[numeric_features].corrwith(cleaner.df['churned'])
    
    print("\n🔗 Feature Correlation with Churn:")
    for feature, corr in correlation_with_churn.sort_values(ascending=False).items():
        if feature != 'churned' and abs(corr) > 0.05:
            print(f"   {feature:30s}: {corr:+.3f}")
    
    # Identify top predictors
    top_predictors = correlation_with_churn.abs().sort_values(ascending=False).head(10)
    print(f"\n🎯 Top 10 churn predictors:")
    for predictor in top_predictors.index:
        if predictor != 'churned':
            print(f"   • {predictor}")
    
    print("\n💾 Step 7: Exporting ML-ready data...")
    os.makedirs('outputs/cleaned_data', exist_ok=True)
    
    # Export full dataset
    cleaner.export_clean_data('outputs/cleaned_data/churn_data_ml_ready.csv')
    
    # Export feature importance
    feature_importance = pd.DataFrame({
        'feature': correlation_with_churn.index,
        'correlation_with_churn': correlation_with_churn.values,
        'abs_correlation': abs(correlation_with_churn.values)
    }).sort_values('abs_correlation', ascending=False)
    
    feature_importance.to_csv('outputs/cleaned_data/feature_importance.csv', index=False)
    
    # Create train/test split
    from utils import split_train_test_by_date
    train_df, test_df = split_train_test_by_date(
        cleaner.df, 
        'last_login_date', 
        test_size=0.3,
        gap_days=7
    )
    
    train_df.to_csv('outputs/cleaned_data/churn_train.csv', index=False)
    test_df.to_csv('outputs/cleaned_data/churn_test.csv', index=False)
    
    print(f"\n✅ Churn prediction data preparation completed!")
    print(f"   Training set: {len(train_df):,} customers")
    print(f"   Test set: {len(test_df):,} customers")
    print(f"   Total features: {cleaner.df.shape[1]}")
    
    return cleaner.df, train_df, test_df

if __name__ == "__main__":
    churn_data, train_data, test_data = run_churn_analysis()
    
    # Quick ML model demonstration
    print("\n🤖 Quick ML Model Performance Check:")
    
    # Prepare features and target
    X_train = train_data.drop(columns=['churned', 'customer_id', 'signup_date', 'last_login_date'])
    y_train = train_data['churned']
    X_test = test_data.drop(columns=['churned', 'customer_id', 'signup_date', 'last_login_date'])
    y_test = test_data['churned']
    
    # Remove any remaining non-numeric columns
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    
    # Train simple model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"   Model Accuracy:  {accuracy:.3f}")
    print(f"   Precision:       {precision:.3f}")
    print(f"   Recall:          {recall:.3f}")
    print(f"   F1-Score:        {f1:.3f}")
    
    print("\n🎉 Data is ready for advanced churn prediction modeling!")
