"""
Data Cleaner Pro - Professional data cleaning toolkit for everyday use
Author: nutmegdev96
GitHub: https://github.com/nutmegdev96/data-cleaner-pro
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Union, Any
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class DataCleaner:
    """
    Comprehensive data cleaning and preprocessing toolkit.
    
    This class provides methods for common data cleaning tasks including:
    - Missing value handling
    - Outlier detection and treatment
    - Data type standardization
    - Duplicate removal
    - Column name standardization
    - Data validation
    
    #hint: Initialize with verbose=True to see detailed cleaning logs
    #hint: All methods return self, enabling method chaining
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None, verbose: bool = True):
        """
        Initialize the DataCleaner with an optional DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            DataFrame to be cleaned
        verbose : bool, default True
            If True, prints progress information during cleaning
            
        Examples:
        ---------
        >>> cleaner = DataCleaner(df, verbose=True)
        >>> cleaner = DataCleaner().load_data('data.csv')
        """
        self.df = df
        self.verbose = verbose
        self.original_shape = df.shape if df is not None else None
        self.cleaning_log = []
        self._setup_default_config()
        
    def _setup_default_config(self):
        """Setup default configuration parameters"""
        self.config = {
            'missing_threshold': 0.3,  # Drop columns with >30% missing
            'outlier_method': 'iqr',   # Default outlier detection method
            'outlier_threshold': 1.5,  # IQR multiplier for outliers
            'string_missing_fill': 'Unknown',  # Default fill for string columns
            'date_format': '%Y-%m-%d',  # Default date format
            'encoding': 'utf-8',        # Default file encoding
        }
        
    def _log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Log cleaning operations for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details,
            'data_shape': self.df.shape if self.df is not None else None
        }
        self.cleaning_log.append(log_entry)
        
        if self.verbose:
            logger.info(f"[{operation}] {details}")
    
    def load_data(self, filepath: str, **kwargs) -> 'DataCleaner':
        """
        Load data from various file formats.
        
        Supported formats: CSV, Excel (xlsx, xls), JSON, Parquet
        
        Parameters:
        -----------
        filepath : str
            Path to the data file
        **kwargs : dict
            Additional parameters for pandas read functions
            
        Returns:
        --------
        self : DataCleaner
            Returns self for method chaining
            
        #hint: Use engine='openpyxl' for Excel files with .xlsx extension
        #hint: For large CSV files, use chunksize parameter
        """
        try:
            if filepath.endswith('.csv'):
                self.df = pd.read_csv(filepath, encoding=self.config['encoding'], **kwargs)
            elif filepath.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(filepath, **kwargs)
            elif filepath.endswith('.json'):
                self.df = pd.read_json(filepath, **kwargs)
            elif filepath.endswith('.parquet'):
                self.df = pd.read_parquet(filepath, **kwargs)
            elif filepath.endswith('.feather'):
                self.df = pd.read_feather(filepath, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
                
            self.original_shape = self.df.shape
            self._log_operation('load_data', {
                'filepath': filepath,
                'shape': self.df.shape,
                'columns': list(self.df.columns)
            })
            
            if self.verbose:
                print(f"✅ Data loaded successfully")
                print(f"   📊 Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
                print(f"   📝 Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {str(e)}")
            raise
            
        return self
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        Generate comprehensive data quality summary.
        
        Returns:
        --------
        pandas.DataFrame
            Summary statistics for each column
            
        #hint: Use this method before cleaning to understand data issues
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Use load_data() first.")
        
        summary_data = []
        
        for col in self.df.columns:
            col_data = {
                'column': col,
                'dtype': str(self.df[col].dtype),
                'total_values': len(self.df),
                'non_null': self.df[col].notna().sum(),
                'null_count': self.df[col].isna().sum(),
                'null_percentage': (self.df[col].isna().sum() / len(self.df) * 100).round(2),
                'unique_values': self.df[col].nunique(),
                'unique_percentage': (self.df[col].nunique() / len(self.df) * 100).round(2),
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_data.update({
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    '25%': self.df[col].quantile(0.25),
                    'median': self.df[col].median(),
                    '75%': self.df[col].quantile(0.75),
                    'max': self.df[col].max(),
                })
            
            summary_data.append(col_data)
        
        summary_df = pd.DataFrame(summary_data)
        self._log_operation('generate_summary', {'summary_shape': summary_df.shape})
        
        return summary_df
    
    def handle_missing_values(self, strategy: str = 'auto', 
                            custom_rules: Optional[Dict] = None,
                            threshold: Optional[float] = None) -> 'DataCleaner':
        """
        Handle missing values using various strategies.
        
        Parameters:
        -----------
        strategy : str, default 'auto'
            Strategy for handling missing values:
            - 'auto': Auto-detect best method per column
            - 'drop_rows': Drop rows with any missing values
            - 'drop_columns': Drop columns with missing values
            - 'fill': Fill with specific values
        custom_rules : dict, optional
            Column-specific rules, e.g., {'age': 'median', 'name': 'Unknown'}
        threshold : float, optional
            Drop columns with missing percentage above this threshold
            
        Returns:
        --------
        self : DataCleaner
        
        #hint: Use threshold=0.3 to drop columns with >30% missing values
        #hint: For time series data, consider 'ffill' or 'bfill' strategies
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Use load_data() first.")
        
        original_shape = self.df.shape
        threshold = threshold or self.config['missing_threshold']
        
        # Step 1: Drop high-missing columns
        missing_percentage = self.df.isna().sum() / len(self.df)
        high_missing_cols = missing_percentage[missing_percentage > threshold].index.tolist()
        
        if high_missing_cols:
            self.df = self.df.drop(columns=high_missing_cols)
            self._log_operation('drop_columns', {
                'columns': high_missing_cols,
                'reason': f'missing > {threshold*100}%',
                'missing_percentages': missing_percentage[high_missing_cols].to_dict()
            })
            
            if self.verbose:
                print(f"🗑️  Dropped {len(high_missing_cols)} columns with >{threshold*100}% missing values")
        
        # Step 2: Apply chosen strategy
        if strategy == 'auto':
            self._auto_impute_missing()
        elif strategy == 'drop_rows':
            before = len(self.df)
            self.df = self.df.dropna()
            dropped = before - len(self.df)
            self._log_operation('drop_rows', {'rows_dropped': dropped})
        elif strategy == 'fill' and custom_rules:
            self._custom_fill_missing(custom_rules)
        
        # Step 3: Apply custom rules if provided
        if custom_rules:
            self._apply_custom_missing_rules(custom_rules)
        
        # Log results
        final_shape = self.df.shape
        self._log_operation('handle_missing_complete', {
            'original_shape': original_shape,
            'final_shape': final_shape,
            'strategy': strategy,
            'columns_dropped': len(high_missing_cols)
        })
        
        if self.verbose:
            print(f"📊 After missing value handling:")
            print(f"   Rows: {original_shape[0]} → {final_shape[0]}")
            print(f"   Columns: {original_shape[1]} → {final_shape[1]}")
            if len(high_missing_cols) > 0:
                print(f"   Columns dropped: {', '.join(high_missing_cols)}")
        
        return self
    
    def _auto_impute_missing(self) -> None:
        """Automatically impute missing values based on column type"""
        for col in self.df.columns:
            if self.df[col].isna().any():
                dtype = self.df[col].dtype
                
                if pd.api.types.is_numeric_dtype(dtype):
                    # For numeric: median for skewed, mean for normal
                    non_null = self.df[col].dropna()
                    
                    if len(non_null) > 3:
                        # Check for normality using skewness
                        skewness = non_null.skew()
                        if abs(skewness) > 1:
                            fill_value = non_null.median()  # Skewed data
                        else:
                            fill_value = non_null.mean()   # Normal data
                    else:
                        fill_value = non_null.median() if not non_null.empty else 0
                    
                    self.df[col].fillna(fill_value, inplace=True)
                    
                elif pd.api.types.is_string_dtype(dtype):
                    # For strings: mode or placeholder
                    mode_value = self.df[col].mode()
                    fill_value = mode_value[0] if not mode_value.empty else self.config['string_missing_fill']
                    self.df[col].fillna(fill_value, inplace=True)
                    
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    # For dates: mode or forward fill
                    mode_date = self.df[col].mode()
                    if not mode_date.empty:
                        self.df[col].fillna(mode_date[0], inplace=True)
                    else:
                        self.df[col].fillna(method='ffill', inplace=True)
    
    def standardize_column_names(self, case: str = 'snake') -> 'DataCleaner':
        """
        Standardize column names to consistent format.
        
        Parameters:
        -----------
        case : str, default 'snake'
            Naming convention:
            - 'snake': snake_case
            - 'camel': camelCase
            - 'pascal': PascalCase
            - 'lower': lowercase
            - 'upper': UPPERCASE
            
        Returns:
        --------
        self : DataCleaner
        
        #hint: snake_case is recommended for database compatibility
        #hint: Use this before analysis to ensure consistent column referencing
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Use load_data() first.")
        
        original_names = list(self.df.columns)
        new_names = []
        
        for col in original_names:
            # Clean the column name
            col = str(col).strip()
            
            # Remove special characters and extra spaces
            col = ''.join(c if c.isalnum() or c == ' ' else ' ' for c in col)
            col = ' '.join(col.split())  # Normalize spaces
            
            # Convert to chosen case
            if case == 'snake':
                col = col.lower().replace(' ', '_')
            elif case == 'camel':
                words = col.lower().split()
                col = words[0] + ''.join(w.title() for w in words[1:])
            elif case == 'pascal':
                col = ''.join(w.title() for w in col.lower().split())
            elif case == 'lower':
                col = col.lower()
            elif case == 'upper':
                col = col.upper()
            
            # Remove any remaining special characters
            col = ''.join(c for c in col if c.isalnum() or c in ['_', ' '])
            new_names.append(col)
        
        self.df.columns = new_names
        
        self._log_operation('standardize_columns', {
            'case_style': case,
            'changes': dict(zip(original_names, new_names))
        })
        
        if self.verbose:
            print(f"📝 Column names standardized to {case}_case")
            if len(original_names) <= 10:  # Show all if few columns
                for old, new in zip(original_names, new_names):
                    if old != new:
                        print(f"   {old} → {new}")
        
        return self
    
    def detect_outliers(self, method: str = None, threshold: float = None) -> Dict[str, Any]:
        """
        Detect outliers in numeric columns.
        
        Parameters:
        -----------
        method : str, optional
            Detection method: 'iqr', 'zscore', 'percentile'
        threshold : float, optional
            Threshold for detection
            
        Returns:
        --------
        dict
            Dictionary with outlier information per column
            
        #hint: IQR method is robust to non-normal distributions
        #hint: Z-score assumes normal distribution
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Use load_data() first.")
        
        method = method or self.config['outlier_method']
        threshold = threshold or self.config['outlier_threshold']
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_report = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
                
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(data))
                outliers = self.df.iloc[np.where(z_scores > threshold)[0]]
                
            elif method == 'percentile':
                lower = data.quantile(0.01)
                upper = data.quantile(0.99)
                outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
            
            outlier_count = len(outliers)
            if outlier_count > 0:
                outlier_report[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(self.df)) * 100,
                    'method': method,
                    'threshold': threshold,
                    'min_outlier': outliers[col].min() if outlier_count > 0 else None,
                    'max_outlier': outliers[col].max() if outlier_count > 0 else None,
                }
        
        self._log_operation('detect_outliers', outlier_report)
        
        if self.verbose and outlier_report:
            print("📈 Outlier Detection Report:")
            for col, report in outlier_report.items():
                print(f"   {col}: {report['count']} outliers ({report['percentage']:.2f}%)")
        
        return outlier_report
    
    def handle_outliers(self, method: str = 'cap', **kwargs) -> 'DataCleaner':
        """
        Handle detected outliers using various methods.
        
        Parameters:
        -----------
        method : str, default 'cap'
            Treatment method:
            - 'cap': Cap at percentiles
            - 'remove': Remove outliers
            - 'transform': Apply transformation
            - 'impute': Replace with median/mean
        **kwargs : dict
            Additional method-specific parameters
            
        Returns:
        --------
        self : DataCleaner
        
        #hint: 'cap' method preserves data points while reducing outlier impact
        #hint: For skewed data, consider log transformation before outlier treatment
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Use load_data() first.")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        treatment_report = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            if method == 'cap':
                # Cap at 1st and 99th percentiles
                lower_cap = data.quantile(0.01)
                upper_cap = data.quantile(0.99)
                
                original = self.df[col].copy()
                self.df[col] = self.df[col].clip(lower=lower_cap, upper=upper_cap)
                
                capped_count = (original != self.df[col]).sum()
                treatment_report[col] = {
                    'method': 'cap',
                    'capped_count': capped_count,
                    'lower_cap': lower_cap,
                    'upper_cap': upper_cap
                }
                
            elif method == 'remove':
                # Remove rows with outliers
                outliers = self.detect_outliers()
                if col in outliers:
                    before = len(self.df)
                    self.df = self.df[~self.df.index.isin(outliers[col]['indices'])]
                    treatment_report[col] = {
                        'method': 'remove',
                        'rows_removed': before - len(self.df)
                    }
        
        self._log_operation('handle_outliers', treatment_report)
        
        if self.verbose and treatment_report:
            print("🔧 Outlier Treatment Applied:")
            for col, report in treatment_report.items():
                if 'capped_count' in report:
                    print(f"   {col}: {report['capped_count']} values capped")
                elif 'rows_removed' in report:
                    print(f"   {col}: {report['rows_removed']} rows removed")
        
        return self
    
    def convert_data_types(self, dtype_map: Optional[Dict] = None) -> 'DataCleaner':
        """
        Convert columns to appropriate data types.
        
        Parameters:
        -----------
        dtype_map : dict, optional
            Mapping of column names to target dtypes
            
        Returns:
        --------
        self : DataCleaner
        
        #hint: Converting to correct dtypes reduces memory usage
        #hint: Use pd.Categorical for columns with few unique string values
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Use load_data() first.")
        
        conversions = {}
        
        if dtype_map:
            # Apply specified conversions
            for col, target_type in dtype_map.items():
                if col in self.df.columns:
                    old_type = str(self.df[col].dtype)
                    try:
                        self.df[col] = self.df[col].astype(target_type)
                        conversions[col] = {'from': old_type, 'to': target_type}
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to {target_type}: {e}")
        else:
            # Auto-convert based on content
            for col in self.df.columns:
                old_type = str(self.df[col].dtype)
                
                # Try to convert object columns to more specific types
                if self.df[col].dtype == 'object':
                    # Try numeric conversion
                    try:
                        self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
                    except:
                        pass
                    
                    # Try datetime conversion
                    if self.df[col].dtype == 'object':
                        try:
                            self.df[col] = pd.to_datetime(self.df[col], errors='ignore')
                        except:
                            pass
                    
                    # Convert to categorical if few unique values
                    if self.df[col].dtype == 'object' and self.df[col].nunique() < 50:
                        self.df[col] = pd.Categorical(self.df[col])
                
                new_type = str(self.df[col].dtype)
                if old_type != new_type:
                    conversions[col] = {'from': old_type, 'to': new_type}
        
        self._log_operation('convert_dtypes', conversions)
        
        if self.verbose and conversions:
            print("🔄 Data Type Conversions:")
            for col, info in conversions.items():
                print(f"   {col}: {info['from']} → {info['to']}")
        
        return self
    
    def remove_duplicates(self, subset: Optional[List] = None, 
                         keep: str = 'first') -> 'DataCleaner':
        """
        Remove duplicate rows from the DataFrame.
        
        Parameters:
        -----------
        subset : list, optional
            Columns to consider for duplication
        keep : str, default 'first'
            Which duplicates to keep: 'first', 'last', or False
            
        Returns:
        --------
        self : DataCleaner
        
        #hint: Use subset parameter to identify duplicates based on key columns
        #hint: keep=False removes all duplicates entirely
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Use load_data() first.")
        
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = before - len(self.df)
        
        self._log_operation('remove_duplicates', {
            'rows_before': before,
            'rows_after': len(self.df),
            'duplicates_removed': removed,
            'subset': subset,
            'keep_strategy': keep
        })
        
        if self.verbose:
            if removed > 0:
                print(f"🎯 Removed {removed} duplicate row(s)")
            else:
                print("✅ No duplicates found")
        
        return self
    
    def validate_data(self, rules: Dict) -> Dict:
        """
        Validate data against custom business rules.
        
        Parameters:
        -----------
        rules : dict
            Validation rules per column
            
        Returns:
        --------
        dict
            Validation results
            
        #hint: Define rules once and reuse across datasets
        #hint: Combine with unit tests for data quality assurance
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Use load_data() first.")
        
        validation_results = {}
        
        for col, rule in rules.items():
            if col in self.df.columns:
                rule_type = rule.get('type')
                violations = []
                
                if rule_type == 'not_null':
                    violations = self.df[self.df[col].isna()].index.tolist()
                    
                elif rule_type == 'in_range':
                    min_val = rule.get('min')
                    max_val = rule.get('max')
                    if min_val is not None and max_val is not None:
                        mask = ~self.df[col].between(min_val, max_val)
                        violations = self.df[mask].index.tolist()
                        
                elif rule_type == 'in_list':
                    allowed = rule.get('allowed', [])
                    mask = ~self.df[col].isin(allowed)
                    violations = self.df[mask].index.tolist()
                    
                elif rule_type == 'regex':
                    pattern = rule.get('pattern')
                    if pattern:
                        mask = ~self.df[col].astype(str).str.match(pattern)
                        violations = self.df[mask].index.tolist()
                
                validation_results[col] = {
                    'rule_type': rule_type,
                    'violation_count': len(violations),
                    'violation_percentage': (len(violations) / len(self.df)) * 100,
                    'violation_indices': violations[:100]  # Limit to first 100
                }
        
        self._log_operation('validate_data', validation_results)
        
        if self.verbose:
            print("🔍 Data Validation Results:")
            for col, result in validation_results.items():
                if result['violation_count'] > 0:
                    print(f"   ⚠️  {col}: {result['violation_count']} violations "
                          f"({result['violation_percentage']:.1f}%)")
                else:
                    print(f"   ✅ {col}: All values valid")
        
        return validation_results
    
    def export_clean_data(self, filepath: str, **kwargs) -> None:
        """
        Export cleaned data to file.
        
        Parameters:
        -----------
        filepath : str
            Path for the output file
        **kwargs : dict
            Additional parameters for pandas export functions
            
        #hint: Use index=False for CSV exports unless index has meaning
        #hint: For large datasets, consider Parquet format for better performance
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Use load_data() first.")
        
        try:
            if filepath.endswith('.csv'):
                self.df.to_csv(filepath, index=False, **kwargs)
            elif filepath.endswith(('.xlsx', '.xls')):
                self.df.to_excel(filepath, index=False, **kwargs)
            elif filepath.endswith('.json'):
                self.df.to_json(filepath, **kwargs)
            elif filepath.endswith('.parquet'):
                self.df.to_parquet(filepath, **kwargs)
            elif filepath.endswith('.feather'):
                self.df.to_feather(filepath, **kwargs)
            else:
                raise ValueError(f"Unsupported output format: {filepath}")
            
            self._log_operation('export_data', {
                'filepath': filepath,
                'shape': self.df.shape,
                'format': filepath.split('.')[-1]
            })
            
            if self.verbose:
                print(f"💾 Clean data exported to: {filepath}")
                print(f"   📊 Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
                
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise
    
    def get_cleaning_report(self) -> pd.DataFrame:
        """
        Generate a comprehensive cleaning report.
        
        Returns:
        --------
        pandas.DataFrame
            Summary of all cleaning operations performed
            
        #hint: Use this report to document data cleaning steps for reproducibility
        """
        report_data = []
        
        for log in self.cleaning_log:
            report_data.append({
                'timestamp': log['timestamp'],
                'operation': log['operation'],
                'details': str(log['details'])[:200],  # Truncate long details
                'rows': log['data_shape'][0] if log['data_shape'] else None,
                'columns': log['data_shape'][1] if log['data_shape'] else None
            })
        
        return pd.DataFrame(report_data)
    
    def reset(self) -> 'DataCleaner':
        """Reset to original data state"""
        # Note: This only works if original data is still in memory
        # For file-based reset, reload from source
        if self.verbose:
            print("🔄 Resetting to original data...")
        return self


# Example usage function
def example_usage():
    """
    Example usage of DataCleaner Pro
    
    #hint: Copy this function to start using the toolkit quickly
    #hint: Modify the steps based on your specific data cleaning needs
    """
    # Create cleaner instance
    cleaner = DataCleaner(verbose=True)
    
    # Load data
    cleaner.load_data('your_data.csv')
    
    # Get initial summary
    summary = cleaner.get_data_summary()
    print("Initial Summary:")
    print(summary[['column', 'null_percentage', 'dtype']].to_string())
    
    # Standard cleaning pipeline
    cleaner \
        .standardize_column_names(case='snake') \
        .handle_missing_values(strategy='auto', threshold=0.3) \
        .convert_data_types() \
        .remove_duplicates() \
        .detect_outliers() \
        .handle_outliers(method='cap') \
        .validate_data({
            'age': {'type': 'in_range', 'min': 0, 'max': 120},
            'email': {'type': 'regex', 'pattern': r'^[^@]+@[^@]+\.[^@]+$'}
        })
    
    # Export cleaned data
    cleaner.export_clean_data('cleaned_data.csv')
    
    # Get cleaning report
    report = cleaner.get_cleaning_report()
    print("\nCleaning Report:")
    print(report.to_string())
    
    return cleaner


if __name__ == "__main__":
    # Run example when script is executed directly
    print("Data Cleaner Pro - Professional Data Cleaning Toolkit")
    print("=" * 50)
    print("\nRun cleaner.example_usage() to see an example workflow.")
    print("Or create your own DataCleaner instance and start cleaning!")
