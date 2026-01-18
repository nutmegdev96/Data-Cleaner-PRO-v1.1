"""
Data Cleaner Pro - Professional data cleaning toolkit
"""

__version__ = '0.1.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'
__license__ = 'MIT'
__description__ = 'Professional data cleaning toolkit for everyday use'

# Import main classes for easier access
from .cleaner import DataCleaner
from .transformers import DataTransformer
from .validators import DataValidator

# Export main classes
__all__ = [
    'DataCleaner',
    'DataTransformer',
    'DataValidator',
    '__version__',
]
