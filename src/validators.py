"""
Data validation utilities for Data Cleaner Pro
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class DataValidator:
    """
    Comprehensive data validation toolkit
    
    #hint: Use this for data quality checks and business rule validation
    #hint: All validators return detailed reports for debugging
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.validation_results = {}
    
    def validate_schema(self, df: pd.DataFrame,
                       expected_schema: Dict[str, type]) -> Dict[str, Any]:
        """
        Validate DataFrame schema against expected types
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to validate
        expected_schema : dict
            Dictionary mapping column names to expected types
            
        Returns:
        --------
        dict
            Validation results
            
        #hint: Use this to ensure data meets pipeline requirements
        """
        results = {
            'total_columns': len(df.columns),
            'matched_columns': 0,
            'mismatched_columns': {},
            'missing_columns': [],
            'extra_columns': []
        }
        
        expected_cols = set(expected_schema.keys())
        actual_cols = set(df.columns)
        
        # Check missing columns
        results['missing_columns'] = list(expected_cols - actual_cols)
        
        # Check extra columns
        results['extra_columns'] = list(actual_cols - expected_cols)
        
        # Check type matches for common columns
        common_cols = expected_cols.intersection(actual_cols)
        
        for col in common_cols:
            expected_type = expected_schema[col]
            actual_type = df[col].dtype
            
            # Convert pandas types to Python types for comparison
            type_mapping = {
                'int64': int, 'float64': float, 'object': str,
                'bool': bool, 'datetime64[ns]': datetime
            }
            
            actual_py_type = type_mapping.get(str(actual_type), type(None))
            
            if expected_type != actual_py_type:
                results['mismatched_columns'][col] = {
                    'expected': expected_type.__name__,
                    'actual': str(actual_type)
                }
            else:
                results['matched_columns'] += 1
        
        self.validation_results['schema'] = results
        
        if self.verbose:
            print("🔍 Schema Validation Results:")
            print(f"   ✅ Matched columns: {results['matched_columns']}/{len(common_cols)}")
            if results['missing_columns']:
                print(f"   ❌ Missing columns: {', '.join(results['missing_columns'])}")
            if results['mismatched_columns']:
                print(f"   ⚠️  Type mismatches: {len(results['mismatched_columns'])}")
        
        return results
    
    def validate_ranges(self, df: pd.DataFrame,
                       range_rules: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Validate numeric columns against range rules
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to validate
        range_rules : dict
            Dictionary mapping columns to min/max rules
            Example: {'age': {'min': 0, 'max': 120}}
            
        Returns:
        --------
        dict
            Validation results
            
        #hint: Useful for business logic validation (e.g., age, price ranges)
        """
        results = {}
        
        for col, rules in range_rules.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                min_val = rules.get('min', -np.inf)
                max_val = rules.get('max', np.inf)
                
                # Find violations
                violations = df[
                    (df[col] < min_val) | 
                    (df[col] > max_val)
                ][col]
                
                results[col] = {
                    'rule': f'{min_val} <= x <= {max_val}',
                    'violations': len(violations),
                    'violation_percentage': (len(violations) / len(df)) * 100,
                    'min_violation': violations.min() if len(violations) > 0 else None,
                    'max_violation': violations.max() if len(violations) > 0 else None,
                    'indices': violations.index.tolist()[:10]  # First 10 only
                }
        
        self.validation_results['ranges'] = results
        
        if self.verbose and results:
            print("📏 Range Validation Results:")
            for col, result in results.items():
                if result['violations'] > 0:
                    print(f"   ⚠️  {col}: {result['violations']} violations "
                          f"({result['violation_percentage']:.1f}%)")
        
        return results
    
    def validate_patterns(self, df: pd.DataFrame,
                         pattern_rules: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate string columns against regex patterns
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to validate
        pattern_rules : dict
            Dictionary mapping columns to regex patterns
            Example: {'email': r'^[^@]+@[^@]+\.[^@]+$'}
            
        Returns:
        --------
        dict
            Validation results
            
        #hint: Use for email, phone number, ID format validation
        """
        results = {}
        
        for col, pattern in pattern_rules.items():
            if col in df.columns:
                # Convert to string and check pattern
                violations = df[
                    ~df[col].astype(str).str.match(pattern, na=False)
                ][col]
                
                results[col] = {
                    'pattern': pattern,
                    'violations': len(violations),
                    'violation_percentage': (len(violations) / len(df)) * 100,
                    'examples': violations.head(5).tolist(),
                    'indices': violations.index.tolist()[:10]
                }
        
        self.validation_results['patterns'] = results
        
        if self.verbose and results:
            print("🔤 Pattern Validation Results:")
            for col, result in results.items():
                if result['violations'] > 0:
                    print(f"   ⚠️  {col}: {result['violations']} violations "
                          f"({result['violation_percentage']:.1f}%)")
        
        return results
    
    def validate_uniqueness(self, df: pd.DataFrame,
                           unique_columns: List[str]) -> Dict[str, Any]:
        """
        Validate uniqueness of columns
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to validate
        unique_columns : list
            Columns that should be unique
            
        Returns:
        --------
        dict
            Validation results
            
        #hint: Use for primary key validation
        """
        results = {}
        
        for col in unique_columns:
            if col in df.columns:
                duplicates = df[df[col].duplicated(keep=False)][col]
                
                results[col] = {
                    'total_unique': df[col].nunique(),
                    'duplicate_count': len(duplicates),
                    'duplicate_percentage': (len(duplicates) / len(df)) * 100,
                    'duplicate_values': duplicates.unique().tolist()[:5],
                    'indices': duplicates.index.tolist()[:10]
                }
        
        self.validation_results['uniqueness'] = results
        
        if self.verbose and results:
            print("🎯 Uniqueness Validation Results:")
            for col, result in results.items():
                if result['duplicate_count'] > 0:
                    print(f"   ⚠️  {col}: {result['duplicate_count']} duplicates "
                          f"({result['duplicate_percentage']:.1f}%)")
        
        return results
    
    def validate_referential_integrity(self, df_main: pd.DataFrame,
                                      df_reference: pd.DataFrame,
                                      foreign_key: str,
                                      primary_key: str) -> Dict[str, Any]:
        """
        Validate referential integrity between DataFrames
        
        Parameters:
        -----------
        df_main : pandas.DataFrame
            DataFrame with foreign key
        df_reference : pandas.DataFrame
            Reference DataFrame with primary key
        foreign_key : str
            Foreign key column in main DataFrame
        primary_key : str
            Primary key column in reference DataFrame
            
        Returns:
        --------
        dict
            Validation results
            
        #hint: Use for database-like referential integrity checks
        """
        results = {}
        
        if foreign_key in df_main.columns and primary_key in df_reference.columns:
            # Find values in foreign key not in primary key
            invalid_values = set(df_main[foreign_key].dropna().unique()) - \
                           set(df_reference[primary_key].dropna().unique())
            
            violations = df_main[df_main[foreign_key].isin(invalid_values)]
            
            results = {
                'foreign_key': foreign_key,
                'primary_key': primary_key,
                'invalid_values_count': len(invalid_values),
                'violation_rows': len(violations),
                'violation_percentage': (len(violations) / len(df_main)) * 100,
                'invalid_values': list(invalid_values)[:10],
                'violation_indices': violations.index.tolist()[:10]
            }
        
        self.validation_results['referential_integrity'] = results
        
        if self.verbose and results['violation_rows'] > 0:
            print("🔗 Referential Integrity Results:")
            print(f"   ⚠️  {results['violation_rows']} violations "
                  f"({results['violation_percentage']:.1f}%)")
            print(f"   Invalid values: {results['invalid_values']}")
        
        return results
    
    def validate_completeness(self, df: pd.DataFrame,
                            required_columns: List[str]) -> Dict[str, Any]:
        """
        Validate that required columns have no null values
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to validate
        required_columns : list
            Columns that should be complete (no nulls)
            
        Returns:
        --------
        dict
            Validation results
        """
        results = {}
        
        for col in required_columns:
            if col in df.columns:
                null_count = df[col].isna().sum()
                
                results[col] = {
                    'null_count': null_count,
                    'null_percentage': (null_count / len(df)) * 100,
                    'completeness_percentage': 100 - (null_count / len(df)) * 100,
                    'indices': df[df[col].isna()].index.tolist()[:10]
                }
        
        self.validation_results['completeness'] = results
        
        if self.verbose and results:
            print("✅ Completeness Validation Results:")
            for col, result in results.items():
                if result['null_count'] > 0:
                    print(f"   ⚠️  {col}: {result['null_count']} null values "
                          f"({result['null_percentage']:.1f}%)")
                else:
                    print(f"   ✓ {col}: 100% complete")
        
        return results
    
    def generate_validation_report(self) -> pd.DataFrame:
        """
        Generate comprehensive validation report
        
        Returns:
        --------
        pandas.DataFrame
            Complete validation report
        """
        report_data = []
        
        for validation_type, results in self.validation_results.items():
            if isinstance(results, dict):
                for key, result in results.items():
                    if isinstance(result, dict):
                        row = {
                            'validation_type': validation_type,
                            'target': key,
                            'status': 'FAIL' if result.get('violations', 0) > 0 
                                      or result.get('null_count', 0) > 0 
                                      or result.get('duplicate_count', 0) > 0 
                                      else 'PASS',
                            'details': str(result)[:200]  # Truncate
                        }
                        report_data.append(row)
        
        return pd.DataFrame(report_data)
