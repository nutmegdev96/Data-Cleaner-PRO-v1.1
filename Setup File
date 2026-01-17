"""
Setup file for Data Cleaner Pro package
"""
from setuptools import setup, find_packages
import os

# Read the contents of README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read version from __init__.py
def get_version():
    with open(os.path.join('src', '__init__.py'), 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip("'\"")
    return '0.1.0'

setup(
    name='data-cleaner-pro',
    version=get_version(),
    author='Your Name',
    author_email='your.email@example.com',
    description='Professional data cleaning toolkit for everyday use',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/data-cleaner-pro',
    project_urls={
        'Documentation': 'https://github.com/yourusername/data-cleaner-pro#readme',
        'Source': 'https://github.com/yourusername/data-cleaner-pro',
        'Tracker': 'https://github.com/yourusername/data-cleaner-pro/issues',
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: OS Independent',
    ],
    keywords=[
        'data-cleaning', 
        'data-preprocessing', 
        'data-analysis', 
        'pandas', 
        'data-science',
        'machine-learning',
        'data-quality'
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.8',
    install_requires=[
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'scipy>=1.9.0',
        'scikit-learn>=1.2.0',
        'python-dateutil>=2.8.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'pre-commit>=3.0.0',
        ],
        'notebooks': [
            'jupyter>=1.0.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'plotly>=5.13.0',
            'ipywidgets>=8.0.0',
        ],
        'docs': [
            'sphinx>=6.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'nbsphinx>=0.9.0',
        ],
        'all': [
            'jupyter>=1.0.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'plotly>=5.13.0',
            'openpyxl>=3.0.0',
            'pyarrow>=10.0.0',
            'sqlalchemy>=1.4.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'datacleaner=data_cleaner_pro.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
