# Quick Start Guide for Data Cleaner Pro

# Import the cleaner
import sys
sys.path.append('../src')
from cleaner import DataCleaner

# 1. Load and explore data
cleaner = DataCleaner(verbose=True)
cleaner.load_data('sample_data.csv')

# 2. Get data summary
summary = cleaner.get_data_summary()
print("Data Summary:")
display(summary.head())

# 3. Standard cleaning pipeline
cleaner.standardize_column_names('snake') \
      .handle_missing_values(strategy='auto') \
      .remove_duplicates() \
      .convert_data_types()

# 4. Detect and handle outliers
outliers = cleaner.detect_outliers()
cleaner.handle_outliers(method='cap')

# 5. Validate data
rules = {
    'age': {'type': 'in_range', 'min': 18, 'max': 100},
    'email': {'type': 'regex', 'pattern': r'^[^@]+@[^@]+\.[^@]+$'}
}
validation = cleaner.validate_data(rules)

# 6. Export clean data
cleaner.export_clean_data('cleaned_sample.csv')

print("✅ Cleaning complete!")
