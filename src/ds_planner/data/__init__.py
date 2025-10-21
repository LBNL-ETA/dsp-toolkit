"""Example datasets for ds_planner.

This module provides easy access to example datasets that demonstrate
the functionality of ds_planner.
"""

import os
from pathlib import Path
import pandas as pd
from typing import Dict, Union, Optional

# Get the directory containing the example data files
DATA_DIR = Path(__file__).parent

def _get_file_description(filepath: Path) -> str:
    """Extract description from the first line of CSV/Parquet files.
    
    Attempts to read the first line of CSV files or metadata from Parquet files
    to use as a description. Falls back to a generic description if unable to
    extract meaningful information.
    
    Args:
        filepath: Path to the data file
    
    Returns:
        String description of the dataset
    """
    try:
        if filepath.suffix == '.csv':
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                # If first line starts with #, treat it as description
                if first_line.startswith('#'):
                    return first_line[1:].strip()
                    
        elif filepath.suffix == '.parquet':
            # Try to read metadata from parquet file
            df = pd.read_parquet(filepath)
            if hasattr(df, '_metadata') and 'description' in df._metadata:
                return df._metadata['description']
                
    except Exception:
        pass
        
    # Fallback to generic description
    return f"Example dataset: {filepath.stem}"

def get_available_datasets() -> Dict[str, Dict[str, str]]:
    """Get a dictionary of available example datasets.
    
    Scans the data directory for CSV and Parquet files and builds a metadata
    dictionary for each discovered dataset.
    
    Returns:
        Dictionary mapping dataset names to their metadata including
        description and file format.
    """
    datasets = {}
    
    # Scan for CSV and Parquet files
    for filepath in DATA_DIR.glob('*.*'):
        if filepath.suffix.lower() in ('.csv', '.parquet'):
            name = filepath.stem
            file_format = filepath.suffix[1:].lower()  # Remove leading dot
            
            datasets[name] = {
                "description": _get_file_description(filepath),
                "format": file_format
            }
    
    return datasets

def load_dataset(
    name: str,
    file_format: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """Load an example dataset.
    
    Args:
        name: Name of the dataset (without extension)
        file_format: Optional file format override ('csv' or 'parquet')
        **kwargs: Additional arguments passed to pd.read_csv or pd.read_parquet
    
    Returns:
        pandas DataFrame containing the dataset
    
    Raises:
        ValueError: If the specified dataset name doesn't exist
        ValueError: If the file format is not supported
        
    Examples:
        >>> from ds_planner.data import load_dataset
        >>> # Load parquet file
        >>> df_parquet = load_dataset("example1")
        >>> # Load CSV file with specific options
        >>> df_csv = load_dataset("example_weather", parse_dates=['timestamp'])
    """
    datasets = get_available_datasets()
    if name not in datasets:
        raise ValueError(
            f"Dataset '{name}' not found. Available datasets: {list(datasets.keys())}"
        )
    
    # Determine file format
    if file_format is None:
        file_format = datasets[name]["format"]
    
    file_path = DATA_DIR / f"{name}.{file_format}"
    
    if file_format == "parquet":
        return pd.read_parquet(file_path, **kwargs)
    elif file_format == "csv":
        return pd.read_csv(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

def get_dataset_path(name: str) -> Path:
    """Get the full path to an example dataset.
    
    Args:
        name: Name of the dataset
    
    Returns:
        Path object pointing to the dataset file
    
    Raises:
        ValueError: If the specified dataset name doesn't exist
    """
    datasets = get_available_datasets()
    if name not in datasets:
        raise ValueError(
            f"Dataset '{name}' not found. Available datasets: {list(datasets.keys())}"
        )
    
    file_format = datasets[name]["format"]
    return DATA_DIR / f"{name}.{file_format}"