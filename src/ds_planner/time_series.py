# src/ds_planner/time_series.py

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.dataprocessing import Pipeline

class DartsTimeSeries:
    def __init__(self, df, freq, target_cols, past_cov_cols=None, future_cov_cols=None, fill_missing=True, target_precision='float32', notes=None):
        self.freq = freq
        self.target_cols = [target_cols] if isinstance(target_cols, str) else target_cols
        self.past_cov_cols = past_cov_cols if past_cov_cols else []
        self.future_cov_cols = future_cov_cols if future_cov_cols else []
        self.target_precision = target_precision
        self.notes = notes

        # Convert float columns to target precision
        float_cols = df.select_dtypes(include=['float64', 'float32']).columns
        df = df.astype({col: getattr(np, self.target_precision) for col in float_cols})

        self.df = df

        # Handle the case where past_cov_cols or future_cov_cols might be empty
        all_cols = self.target_cols + self.past_cov_cols + self.future_cov_cols
        self.ts = TimeSeries.from_dataframe(
            df[all_cols], 
            freq=freq
        )
        self.valid_cols = all_cols
        
        # Add holidays only if future_cov_cols is not empty
        if self.future_cov_cols:
            self.ts = self.ts.add_holidays(country_code='US')
        
        if fill_missing:
            self.fill_missing_values()
            
        # Cast ts to target_precision
        self.ts = self.ts.astype(np.float32)

        # Prepare target and covariate data
        self.targets = self.ts[self.target_cols]
        self.past_covariates = self.ts[self.past_cov_cols] if self.past_cov_cols else None
        self.future_covariates = self.ts[self.future_cov_cols + ['holidays']] if self.future_cov_cols else None

        # Define separate pipelines
        self.target_pipeline = Pipeline([Scaler()])
        self.past_cov_pipeline = Pipeline([Scaler()]) if self.past_cov_cols else None
        self.future_cov_pipeline = Pipeline([Scaler()]) if self.future_cov_cols else None

        # Fit and transform data using the respective pipelines
        self.piped_targets = self.target_pipeline.fit_transform(self.targets)
        self.piped_past_covariates = self.past_cov_pipeline.fit_transform(self.past_covariates) if self.past_cov_pipeline else None
        self.piped_future_covariates = self.future_cov_pipeline.fit_transform(self.future_covariates) if self.future_cov_pipeline else None

        # Create a scaled TimeSeries object
        self.all_col_pipeline = Pipeline([Scaler()])
        self.ts_scaled = self.all_col_pipeline.fit_transform(self.ts)

    def __str__(self):
        return f"DartsTimeSeries with {len(self)} samples, {len(self.ts.columns)} features from {self.df.index[0]} to {self.df.index[-1]} at frequency {self.freq}"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Handle slice objects
            start, stop, step = key.indices(len(self))
            new_df = self.df.iloc[start:stop:step]
        elif isinstance(key, (int, np.integer)):
            # Handle integer indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("Index out of range")
            new_df = self.df.iloc[[key]]
        else:
            raise TypeError("Invalid argument type: {}".format(type(key)))

        new_instance = DartsTimeSeries(
            new_df, 
            self.freq, 
            self.target_cols,
            self.past_cov_cols,
            self.future_cov_cols,
            fill_missing=False,
            target_precision=self.target_precision
        )

        # Slice the TimeSeries objects
        new_instance.ts = self.ts[key]
        new_instance.targets = self.targets[key]
        new_instance.past_covariates = self.past_covariates[key] if self.past_covariates is not None else None
        new_instance.future_covariates = self.future_covariates[key] if self.future_covariates is not None else None

        # Slice the piped data
        new_instance.piped_targets = self.piped_targets[key]
        new_instance.piped_past_covariates = self.piped_past_covariates[key] if self.piped_past_covariates is not None else None
        new_instance.piped_future_covariates = self.piped_future_covariates[key] if self.piped_future_covariates is not None else None

        # Keep the same pipelines
        new_instance.target_pipeline = self.target_pipeline
        new_instance.past_cov_pipeline = self.past_cov_pipeline
        new_instance.future_cov_pipeline = self.future_cov_pipeline

        return new_instance

    def fill_missing_values(self):
        filler = MissingValuesFiller()
        self.ts = filler.transform(self.ts)

    def get_precision(self):
        return self.target_precision

