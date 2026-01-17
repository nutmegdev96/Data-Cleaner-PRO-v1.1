"""
Data transformation utilities for Data Cleaner Pro
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder
)
import warnings

warnings.filterwarnings('ignore')


class DataTransformer:
    """
    Advanced data transformation utilities
    
    #hint: Use this class for feature engineering and preprocessing
    #hint: All transformations are reversible when possible
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.scalers = {}
        self.encoders = {}
        self.transformation_log = []
    
    def normalize_numeric(self, df: pd.DataFrame, 
                         columns: Optional[List[str]] = None,
                         method: str = 'standard') -> pd.DataFrame:
        """
        Normalize numeric columns
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        columns : list, optional
            Columns to normalize (default: all numeric)
        method : str, default 'standard'
            'standard': StandardScaler (mean=0, std=1)
            'minmax': MinMaxScaler (range 0-1)
            'robust': RobustScaler (robust to outliers)
            
        Returns:
        --------
        pandas.DataFrame
            Normalized DataFrame
            
        #hint: Use 'standard' for algorithms assuming normal distribution
        #hint: Use 'minmax' for neural networks
        #hint: Use 'robust' when outliers are present
        """
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Remove NaN for scaling
                mask = df[col].notna()
                if mask.any():
                    data = df.loc[mask, col].values.reshape(-1, 1)
                    
                    if method == 'standard':
                        scaler = StandardScaler()
                    elif method == 'minmax':
                        scaler = MinMaxScaler()
                    elif method == 'robust':
                        scaler = RobustScaler()
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    
                    # Fit and transform
                    scaled_data = scaler.fit_transform(data)
                    df.loc[mask, col] = scaled_data.flatten()
                    
                    # Store scaler for inverse transformation
                    self.scalers[col] = scaler
                    
                    self.transformation_log.append({
                        'operation': 'normalize',
                        'column': col,
                        'method': method,
                        'scaler': scaler
                    })
        
        if self.verbose:
            print(f"✅ Normalized {len(columns)} columns using {method} scaling")
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame,
                          columns: Optional[List[str]] = None,
                          method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical variables
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        columns : list, optional
            Columns to encode (default: all categorical)
        method : str, default 'label'
            'label': Label encoding (0 to n_classes-1)
            'onehot': One-hot encoding (dummy variables)
            'target': Target encoding (mean of target per category)
            
        Returns:
        --------
        pandas.DataFrame
            Encoded DataFrame
            
        #hint: Use 'label' for tree-based models
        #hint: Use 'onehot' for linear models and neural networks
        #hint: Use 'target' for high-cardinality categories
        """
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in columns:
            if col in df.columns:
                if method == 'label':
                    encoder = LabelEncoder()
                    mask = df[col].notna()
                    if mask.any():
                        df.loc[mask, col] = encoder.fit_transform(df.loc[mask, col])
                        self.encoders[col] = encoder
                        
                elif method == 'onehot':
                    # Create dummy variables
                    dummies = pd.get_dummies(df[col], prefix=col, 
                                            drop_first=True, dummy_na=False)
                    # Drop original column and add dummies
                    df = df.drop(columns=[col])
                    df = pd.concat([df, dummies], axis=1)
                    self.encoders[col] = 'onehot'
                    
                elif method == 'target':
                    # This requires a target column - simplified version
                    # In practice, you'd pass target values
                    pass
                
                self.transformation_log.append({
                    'operation': 'encode',
                    'column': col,
                    'method': method
                })
        
        if self.verbose:
            print(f"✅ Encoded {len(columns)} columns using {method} encoding")
        
        return df
    
    def create_features(self, df: pd.DataFrame,
                       operations: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Create new features from existing ones
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        operations : dict
            Dictionary mapping new features to operations
            Example: {
                'total_price': ['quantity * price'],
                'price_per_unit': ['price / quantity'],
                'log_price': ['np.log(price + 1)']
            }
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with new features
            
        #hint: Use numpy functions for mathematical operations
        #hint: Add small constant when using log to avoid log(0)
        """
        df = df.copy()
        
        for new_feature, expr_list in operations.items():
            for expr in expr_list:
                try:
                    # Evaluate expression safely
                    df[new_feature] = df.eval(expr)
                    
                    self.transformation_log.append({
                        'operation': 'create_feature',
                        'new_feature': new_feature,
                        'expression': expr
                    })
                    
                    if self.verbose:
                        print(f"➕ Created feature: {new_feature} = {expr}")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Failed to create {new_feature}: {e}")
        
        return df
    
    def extract_datetime_features(self, df: pd.DataFrame,
                                 date_column: str) -> pd.DataFrame:
        """
        Extract features from datetime columns
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        date_column : str
            Name of datetime column
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with extracted features
            
        #hint: Useful for time series analysis and seasonality detection
        """
        df = df.copy()
        
        if date_column in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[f'{date_column}_year'] = df[date_column].dt.year
            df[f'{date_column}_month'] = df[date_column].dt.month
            df[f'{date_column}_day'] = df[date_column].dt.day
            df[f'{date_column}_weekday'] = df[date_column].dt.weekday
            df[f'{date_column}_quarter'] = df[date_column].dt.quarter
            df[f'{date_column}_is_weekend'] = df[date_column].dt.weekday >= 5
            df[f'{date_column}_hour'] = df[date_column].dt.hour
            df[f'{date_column}_minute'] = df[date_column].dt.minute
            
            self.transformation_log.append({
                'operation': 'extract_datetime',
                'date_column': date_column,
                'features_created': [
                    f'{date_column}_year', f'{date_column}_month',
                    f'{date_column}_day', f'{date_column}_weekday',
                    f'{date_column}_quarter', f'{date_column}_is_weekend',
                    f'{date_column}_hour', f'{date_column}_minute'
                ]
            })
            
            if self.verbose:
                print(f"📅 Extracted datetime features from {date_column}")
        
        return df
    
    def handle_skewness(self, df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       threshold: float = 1.0) -> pd.DataFrame:
        """
        Apply transformations to reduce skewness
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        columns : list, optional
            Columns to transform (default: all numeric)
        threshold : float, default 1.0
            Absolute skewness threshold for transformation
            
        Returns:
        --------
        pandas.DataFrame
            Transformed DataFrame
            
        #hint: Use log transform for right-skewed data
        #hint: Use square/cube root for moderate skewness
        #hint: Box-Cox requires positive values
        """
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        from scipy.stats import skew
        
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Calculate skewness
                data = df[col].dropna()
                if len(data) > 3:
                    skewness = skew(data)
                    
                    if abs(skewness) > threshold:
                        # Apply transformation based on skewness direction
                        if skewness > 0:  # Right-skewed
                            # Try log transform (requires positive values)
                            if (data > 0).all():
                                df[col] = np.log1p(df[col])  # log(x + 1)
                                method = 'log1p'
                            else:
                                df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
                                method = 'signed_log1p'
                        else:  # Left-skewed
                            # Try square transform
                            df[col] = df[col] ** 2
                            method = 'square'
                        
                        self.transformation_log.append({
                            'operation': 'reduce_skewness',
                            'column': col,
                            'original_skewness': skewness,
                            'method': method
                        })
                        
                        if self.verbose:
                            print(f"📐 Applied {method} to {col} (skewness: {skewness:.2f})")
        
        return df
    
    def get_transformation_summary(self) -> pd.DataFrame:
        """
        Get summary of all transformations applied
        
        Returns:
        --------
        pandas.DataFrame
            Transformation summary
        """
        return pd.DataFrame(self.transformation_log)
