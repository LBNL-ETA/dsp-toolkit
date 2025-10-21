"""Utility functions for ds_planner package.

This module provides utility functions for data preprocessing and feature engineering,
particularly focused on time series data manipulation.
"""

from typing import Optional, Union, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from darts import TimeSeries


def check_create_path(path: Union[str, Path], exist_ok: bool = True) -> None:
    """Check if a path exists and create it if it doesn't.
    
    Args:
        path: Path to check/create. Can be either a string or Path object.
        exist_ok: If False, raise an error if path already exists.
            If True, silently complete if path exists. Default: True.
    
    Raises:
        FileExistsError: If exist_ok is False and path already exists.
        PermissionError: If the program lacks permission to create the directory.
    
    Examples:
        >>> check_create_path("/path/to/directory")
        Path created: /path/to/directory
        
        >>> check_create_path("existing/path", exist_ok=False)
        FileExistsError: Path already exists: existing/path
    """
    path = Path(path)
    try:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=exist_ok)
            print(f"Path created: {path}")
        else:
            if not exist_ok:
                raise FileExistsError(f"Path already exists: {path}")
            print(f"Path exists: {path}")
            
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied when creating directory: {path}"
        ) from e

def seasonal_split(
    dts: TimeSeries,
    n_chunks: int = 4,
    val_ratio: float = 0.2
) -> Tuple[List[TimeSeries], List[TimeSeries]]:
    """Split a time series into training and validation sets using seasonal chunks.
    
    This function splits the time series into chunks and then splits each chunk
    into training and validation sets according to the specified ratio. This
    approach helps maintain the seasonal patterns in both training and validation
    sets.
    
    Args:
        dts: Input time series to split
        n_chunks: Number of chunks to split the series into. Default: 4
        val_ratio: Ratio of validation data in each chunk (0 to 1). Default: 0.2
    
    Returns:
        A tuple containing:
            - List of training time series chunks
            - List of validation time series chunks
    
    Raises:
        ValueError: If n_chunks < 1 or val_ratio is not between 0 and 1
        ValueError: If the time series is too short for the specified split
    
    Examples:
        >>> ts = TimeSeries.from_dataframe(df)
        >>> train_series, val_series = seasonal_split(ts, n_chunks=4, val_ratio=0.2)
        >>> print(f"Number of training chunks: {len(train_series)}")
        Number of training chunks: 4
    """
    # Input validation
    if n_chunks < 1:
        raise ValueError(f"n_chunks must be >= 1, got {n_chunks}")
    
    if not 0 <= val_ratio <= 1:
        raise ValueError(
            f"val_ratio must be between 0 and 1, got {val_ratio}"
        )
    
    # Calculate chunk size
    total_length = len(dts)
    chunk_size = total_length // n_chunks
    
    if chunk_size < 2:
        raise ValueError(
            f"Time series too short ({total_length} points) for {n_chunks} chunks"
        )
    
    train_series: List[TimeSeries] = []
    val_series: List[TimeSeries] = []
    
    # Split series into chunks
    for i in range(n_chunks):
        start = i * chunk_size
        # Handle the last chunk which might be larger
        end = (i + 1) * chunk_size if i < n_chunks - 1 else total_length
        chunk = dts[start:end]
        
        # Split each chunk into train and validation
        split_point = int(len(chunk) * (1 - val_ratio))
        if split_point == 0:
            raise ValueError(
                f"Validation ratio {val_ratio} too high for chunk size {len(chunk)}"
            )
        
        train_series.append(chunk[:split_point])
        val_series.append(chunk[split_point:])
    
    return train_series, val_series

def _normalize_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Normalize angle to be between 0 and 2π.
    
    Args:
        angle: Angle in radians.
    
    Returns:
        Normalized angle in radians.
    """
    return angle % (2 * np.pi)

def add_cyclic_features(
    df: pd.DataFrame,
    time_column: Optional[str] = None,
    features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Add cyclic features for temporal data.
    
    Creates cyclic (sine and cosine) features for hour of day, day of week,
    and month of year from datetime information. These features are useful for
    capturing periodic patterns in time series data.
    
    Args:
        df: Input dataframe containing temporal data.
        time_column: Name of the timestamp column. If None, uses the DataFrame index.
        features: List of features to add. Valid options are ['hour', 'day', 'month'].
            If None, adds all features.
    
    Returns:
        DataFrame with added cyclic features. Original DataFrame remains unchanged.
    
    Raises:
        ValueError: If the time_column or index is not in datetime format.
        ValueError: If features contains invalid feature names.
        TypeError: If input df is not a pandas DataFrame.
    
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'time': pd.date_range('2023-01-01', periods=24, freq='H')})
        >>> df_cyclic = add_cyclic_features(df, time_column='time', features=['hour'])
        >>> print(df_cyclic.columns)
        Index(['time', 'hour_sin', 'hour_cos'])
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")
    
    # Make a deep copy to avoid modifying the input
    df_out = df.copy(deep=True)
    
    # Validate and process time column/index
    if time_column is None:
        time_series = df_out.index
    else:
        if time_column not in df_out.columns:
            raise ValueError(f"Column '{time_column}' not found in DataFrame")
        time_series = pd.to_datetime(df_out[time_column], errors='coerce')
        
    # Validate datetime format
    if not isinstance(time_series, pd.DatetimeIndex):
        try:
            time_series = pd.DatetimeIndex(time_series)
        except (ValueError, TypeError):
            raise ValueError(
                "The index or specified time column must be convertible to datetime format"
            )
    
    # Validate features parameter
    valid_features = {'hour', 'day', 'month'}
    if features is None:
        features = list(valid_features)
    else:
        invalid_features = set(features) - valid_features
        if invalid_features:
            raise ValueError(
                f"Invalid features: {invalid_features}. "
                f"Valid options are: {valid_features}"
            )
    
    # Add hour features
    if 'hour' in features:
        hours_in_day = 24
        angle = _normalize_angle(2 * np.pi * time_series.hour / hours_in_day)
        df_out['hour_sin'] = np.sin(angle)
        df_out['hour_cos'] = np.cos(angle)
    
    # Add day of week features
    if 'day' in features:
        days_in_week = 7
        angle = _normalize_angle(2 * np.pi * time_series.dayofweek / days_in_week)
        df_out['day_of_week_sin'] = np.sin(angle)
        df_out['day_of_week_cos'] = np.cos(angle)
    
    # Add month features
    if 'month' in features:
        months_in_year = 12
        angle = _normalize_angle(2 * np.pi * time_series.month / months_in_year)
        df_out['month_sin'] = np.sin(angle)
        df_out['month_cos'] = np.cos(angle)
    
    return df_out


