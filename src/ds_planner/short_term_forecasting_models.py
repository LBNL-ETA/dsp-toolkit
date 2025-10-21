"""Short-term forecasting models for distribution substation planning.

This module provides implementations of forecasting models specifically designed
for short-term load and DER forecasting in distribution substations.
"""

from typing import Optional, Union, Dict, List, Any
import os
import warnings
import inspect
from datetime import datetime

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from darts import TimeSeries
from darts.models import (
    NaiveMean, NaiveSeasonal, NaiveMovingAverage, NaiveDrift,
    ARIMA, AutoARIMA, LinearRegressionModel,
)
from .utils import check_create_path, seasonal_split


class Model:
    """A wrapper class for various short-term time series forecasting models.
    
    This class provides a unified interface for training and evaluating different
    forecasting models, with support for both traditional statistical methods and
    deep learning approaches.
    
    Args:
        model_handle: Name of the model to use (must be in model_configs)
        dev_dts: Development (training and validation) dataset as DartsTimeSeries object
        test_dts: Test dataset as DartsTimeSeries object (optional)
        lookback: Number of past time steps to consider (default: 24)
        horizon: Number of future time steps to predict (default: 6)
        n_chunks: Number of chunks for training data splitting (default: 4)
        val_ratio: Ratio of validation data (default: 0.2)
        n_epochs: Number of training epochs for deep learning models (default: 1)
        max_samples_per_ts: Maximum samples per time series (optional)
        emulator_name: Custom name for the model (optional)
        gpu_id: GPU device ID to use (default: 0)
        work_dir: Working directory for saving models and logs (optional)
        try_early_stop: Whether to use early stopping (default: True)
        dataloader_kwargs: Additional arguments for data loader (optional)
        pl_trainer_kwargs: Additional arguments for PyTorch Lightning trainer (optional)
        **model_kwargs: Additional arguments passed to the model constructor
    
    Example:
        >>> from ds_planner.short_term_forecasting_models import Model
        >>> model = Model(
        ...     model_handle="TFTModel",
        ...     dev_dts=development_data,
        ...     lookback=24,
        ...     horizon=6
        ... )
        >>> model.fit()
        >>> predictions = model.historical_forecasts()
    """

    def __init__(
        self,
        model_handle: str,
        dev_dts: Any,  # DartsTimeSeries object for development (training + validation) data
        test_dts: Optional[Any] = None,  # DartsTimeSeries object for test data
        lookback: int = 24,
        horizon: int = 6,
        n_chunks: int = 4,
        val_ratio: float = 0.2,
        n_epochs: int = 1,
        max_samples_per_ts: Optional[int] = None,
        emulator_name: Optional[str] = None,
        gpu_id: int = 0,
        work_dir: Optional[str] = None,
        try_early_stop: bool = True,
        dataloader_kwargs: Optional[Dict] = None,
        pl_trainer_kwargs: Optional[Dict] = None,
        **model_kwargs: Any
    ) -> None:
        """Initialize the Model."""
        # Store initialization parameters
        self.model_handle = model_handle
        self.dev_dts = dev_dts
        self.test_dts = test_dts
        self.lookback = lookback
        self.horizon = self.test_horizon = horizon
        self.n_chunks = n_chunks
        self.val_ratio = val_ratio
        self.n_epochs = n_epochs
        self.gpu_id = gpu_id
        self.max_samples_per_ts = max_samples_per_ts
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.try_early_stop = try_early_stop
        self.pl_trainer_kwargs = pl_trainer_kwargs
        self.model_kwargs = model_kwargs

        # Set model name
        if emulator_name is None:
            if len(self.dev_dts.target_cols) == 1:
                tgt_cleaned = self.dev_dts.target_cols[0].replace(' ', '-')
            else:
                tgt_cleaned = f"{len(self.dev_dts.target_cols)}ts"
            tag = f"b={lookback}_h={horizon}"
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.emulator_name = (
                f"{timestamp}@{model_handle}_spatial={self.dev_dts.notes}"
                f"_target={tgt_cleaned}_{tag}_gpu={self.gpu_id}"
            )
        else:
            self.emulator_name = emulator_name

        # Configure paths
        self._configure_paths(work_dir)
        
        # Initialize model configurations
        self._init_model_configs()
        
        # Initialize the model
        self._initialize_model()

    def _configure_paths(self, work_dir: Optional[str]) -> None:
        """Configure working directories and paths."""
        self.work_dir = work_dir or os.getcwd()
        self.model_save_path = os.path.join(self.work_dir, "trained_models")
        self.log_path = os.path.join(self.work_dir, "logs")
        self.test_results_path = os.path.join(self.work_dir, "final_test_results")
        self.trained_model_save_path = os.path.join(
            self.model_save_path, f"{self.emulator_name}.pkl"
        )

        # Create necessary directories
        for path in [self.work_dir, self.model_save_path, 
                    self.log_path, self.test_results_path]:
            check_create_path(path)

    def _initialize_model(self) -> None:
        """Initialize the forecasting model with appropriate configuration."""
        config = self.model_configs[self.model_handle]
        init_params = config['init_params'].copy()

        # Check for user-provided parameters
        auto_params = {'lags', 'lags_past_covariates', 
                      'lags_future_covariates', 'output_chunk_length', 
                      'input_chunk_length'}
        if any(param in self.model_kwargs for param in auto_params):
            warnings.warn(
                "User provided lags or horizon-related parameters. "
                "Automatic configuration will be ignored."
            )
        else:
            init_params.update(config['automatic_params'])

        init_params.update(self.model_kwargs)
        self.model = self.model_class(**init_params)
        self.config = config
        
    def _init_model_configs(self) -> None:
        """Initialize model configurations dictionary."""
        self.model_configs: Dict[str, Dict[str, Any]] = {
            'NaiveMean': {
                'model_class': 'NaiveMean',
                'init_params': {},
                'automatic_params': {},
                'supports_past_covariates': False,
                'supports_future_covariates': False,
                'historical_forecast_params': {'retrain': True},
                'supports_pl_trainer': False,
            },
            'NaiveMovingAverage': {
                'model_class': 'NaiveMovingAverage',
                'init_params': {},
                'automatic_params': {'input_chunk_length': self.lookback},
                'supports_past_covariates': False,
                'supports_future_covariates': False,
                'historical_forecast_params': {'retrain': True},
                'supports_pl_trainer': False,
            },
            'LinearRegressionModel_multi-model': {
                'model_class': 'LinearRegressionModel',
                'init_params': {
                    'multi_models': True
                },
                'automatic_params': {
                    'lags': self.lookback,
                    'lags_past_covariates': self.lookback,
                    'lags_future_covariates': (self.lookback, self.horizon),
                    'output_chunk_length': self.horizon
                },
                'supports_past_covariates': True,
                'supports_future_covariates': True,
                'historical_forecast_params': {'retrain': False},
                'supports_pl_trainer': False,
            },
            'LinearRegressionModel_autoregressive': {
                'model_class': 'LinearRegressionModel',
                'init_params': {
                    'multi_models': False
                },
                'automatic_params': {
                    'lags': self.lookback,
                    'lags_past_covariates': self.lookback,
                    'lags_future_covariates': (self.lookback, 1),
                    'output_chunk_length': 1
                },
                'supports_past_covariates': True,
                'supports_future_covariates': True,
                'historical_forecast_params': {'retrain': False},
                'supports_pl_trainer': False,
            },
            'RandomForest_multi-model': {
                'model_class': 'RandomForest',
                'init_params': {
                    'n_estimators': 80,
                    'n_jobs': 48,
                    'multi_models': True
                },
                'automatic_params': {
                    'lags': self.lookback,
                    'lags_past_covariates': self.lookback,
                    'lags_future_covariates': (self.lookback, self.horizon),
                    'output_chunk_length': self.horizon
                },
                'supports_past_covariates': True,
                'supports_future_covariates': True,
                'historical_forecast_params': {'retrain': False},
                'supports_pl_trainer': False,
            },
            'RandomForest_autoregressive': {
                'model_class': 'RandomForest',
                'init_params': {
                    'n_estimators': 80,
                    'n_jobs': 48,
                    'multi_models': False
                },
                'automatic_params': {
                    'lags': self.lookback,
                    'lags_past_covariates': self.lookback,
                    'lags_future_covariates': (self.lookback, 96),
                    'output_chunk_length': 96
                },
                'supports_past_covariates': True,
                'supports_future_covariates': True,
                'historical_forecast_params': {'retrain': False},
                'supports_pl_trainer': False,
            },
            'LightGBMModel' : {
                'model_class': 'LightGBMModel',
                'init_params': {},
                'automatic_params': {
                    'lags': self.lookback,
                    'lags_past_covariates': self.lookback,
                    'lags_future_covariates': (self.lookback, self.horizon),
                    'output_chunk_length': self.horizon
                },
                'supports_past_covariates': True,
                'supports_future_covariates': True,
                'historical_forecast_params': {'retrain': False},
                'supports_pl_trainer': False,
            },
            'XGBModel_multi-model': {
                'model_class': 'XGBModel',
                'init_params': {
                    # 'num_parallel_tree': 1,
                    'n_estimators': 40,
                    'n_jobs': 24,
                    'multi_models': True,
                },
                'automatic_params': {
                    'lags': self.lookback,
                    'lags_past_covariates': self.lookback,
                    'lags_future_covariates': (self.lookback, self.horizon),
                    'output_chunk_length': self.horizon
                },
                'supports_past_covariates': True,
                'supports_future_covariates': True,
                'historical_forecast_params': {'retrain': False},
                'supports_pl_trainer': False,
            },
            'XGBModel_autoregressive': {
                'model_class': 'XGBModel',
                'init_params': {
                    # 'num_parallel_tree': 1,
                    'n_estimators': 40,
                    'n_jobs': 24,
                    'multi_models': False,
                },
                'automatic_params': {
                    'lags': self.lookback,
                    'lags_past_covariates': self.lookback,
                    'lags_future_covariates': (self.lookback, 1),
                    'output_chunk_length': 1
                },
                'supports_past_covariates': True,
                'supports_future_covariates': True,
                'historical_forecast_params': {'retrain': False},
                'supports_pl_trainer': False,
            },
            'ARIMA': {
                'model_class': 'ARIMA',
                'init_params': {
                    'p': 24,
                    'd': 1,
                    'q': 12,
                },
                'automatic_params': {

                },
                'supports_past_covariates': False,
                'supports_future_covariates': True,
                'historical_forecast_params': {'retrain': False},
                'supports_pl_trainer': False,
            },
            'AutoARIMA': {
                'model_class': 'AutoARIMA',
                'init_params': {
                },
                'automatic_params': {
                },
                'supports_past_covariates': False,
                'supports_future_covariates': True,
                'historical_forecast_params': {'retrain': True},
                'supports_pl_trainer': False,
            },
            # Auto-regressive RNN (deterministic)
            'RNNModel': {
                'model_class': 'RNNModel',
                'init_params': {
                    'model': 'LSTM',
                    'batch_size': 128,
                    'hidden_dim': 256,
                    'n_rnn_layers': 3,
                    'dropout': 0.3,
                    'optimizer_kwargs': {'lr': 1e-3, 'weight_decay': 1e-5},
                    'lr_scheduler_cls': CosineAnnealingLR,
                    'lr_scheduler_kwargs': {'T_max': self.n_epochs, 'eta_min': 1e-4},
                    'model_name': self.emulator_name,
                },
                'automatic_params': {
                    'input_chunk_length': self.lookback,
                    'output_chunk_length': self.horizon
                },
                'supports_past_covariates': False,
                'supports_future_covariates': True,
                'historical_forecast_params': {'retrain': False},
                'supports_pl_trainer': True,
            },
            # Seq2Seq RNN with fc output layer
            'BlockRNNModel': {
                'model_class': 'BlockRNNModel',
                'init_params': {
                    'model': 'LSTM',
                    'batch_size': 128,
                    'hidden_dim': 256,
                    'n_rnn_layers': 3,
                    'dropout': 0.3,
                    'optimizer_kwargs': {'lr': 1e-3, 'weight_decay': 1e-5},
                    'lr_scheduler_cls': CosineAnnealingLR,
                    'lr_scheduler_kwargs': {'T_max': self.n_epochs, 'eta_min': 1e-4},
                    'model_name': self.emulator_name,
                },
                'automatic_params': {
                    'input_chunk_length': self.lookback,
                    'output_chunk_length': self.horizon
                },
                'supports_past_covariates': True,
                'supports_future_covariates': False,
                'historical_forecast_params': {'retrain': False},
                'supports_pl_trainer': True,
            },
            'TCNModel': {
                'model_class': 'TCNModel',
                'init_params': {
                    'batch_size': 128,
                    'kernel_size': 3,
                    'num_layers': 3,
                    'dropout': 0.25,
                    'optimizer_kwargs': {'lr': 1e-3, 'weight_decay': 1e-5},
                    'lr_scheduler_cls': CosineAnnealingLR,
                    'lr_scheduler_kwargs': {'T_max': self.n_epochs, 'eta_min': 1e-4},
                    'model_name': self.emulator_name,
                },
                'automatic_params': {
                    'input_chunk_length': self.lookback,
                    'output_chunk_length': self.horizon
                },
                'supports_past_covariates': True,
                'supports_future_covariates': False,
                'historical_forecast_params': {'retrain': False},
                'supports_pl_trainer': True,
            },
            'TFTModel': {
                'model_class': 'TFTModel',
                'init_params': {
                    'batch_size': 128,
                    'hidden_size': 192,
                    'num_attention_heads': 4,
                    'lstm_layers': 2,
                    'dropout': 0.25,
                    'optimizer_kwargs': {'lr': 1e-3, 'weight_decay': 1e-5},
                    'lr_scheduler_cls': CosineAnnealingLR,
                    'lr_scheduler_kwargs': {'T_max': self.n_epochs, 'eta_min': 1e-4},
                    'model_name': self.emulator_name,
                },
                'automatic_params': {
                    'input_chunk_length': self.lookback,
                    'output_chunk_length': self.horizon
                },
                'supports_past_covariates': True,
                'supports_future_covariates': True,
                'historical_forecast_params': {'retrain': False},
                'supports_pl_trainer': True,
            },
            'NHiTSModel': {
                'model_class': 'NHiTSModel',
                'init_params': {
                    'batch_size': 256,
                    'dropout': 0.1,
                    'num_blocks': 3,
                    'num_layers': 3,
                    'layer_widths': 512,
                    'optimizer_kwargs': {'lr': 1e-3, 'weight_decay': 1e-5},
                    'lr_scheduler_cls': CosineAnnealingWarmRestarts,
                    'lr_scheduler_kwargs': {'T_0': 5, 'T_mult': 2, 'eta_min': 1e-5},
                    'model_name': self.emulator_name,
                },
                'automatic_params': {
                    'input_chunk_length': self.lookback,
                    'output_chunk_length': self.horizon
                },
                'supports_past_covariates': True,
                'supports_future_covariates': False,
                'historical_forecast_params': {'retrain': False},
                'supports_pl_trainer': True,
            },
            'NBEATSModel': {
                'model_class': 'NBEATSModel',
                'init_params': {
                    'optimizer_kwargs': {'lr': 1e-3, 'weight_decay': 1e-5},
                    'lr_scheduler_cls': CosineAnnealingLR,
                    'lr_scheduler_kwargs': {'T_max': self.n_epochs, 'eta_min': 1e-4},
                    'model_name': self.emulator_name,
                },
                'automatic_params': {
                    'input_chunk_length': self.lookback,
                    'output_chunk_length': self.horizon
                },
                'supports_past_covariates': True,
                'supports_future_covariates': False,
                'historical_forecast_params': {'retrain': False},
                'supports_pl_trainer': True,
            },
            'TSMixerModel': {
                'model_class': 'TSMixerModel',
                'init_params': {
                    'dropout': 0.1,
                    'optimizer_kwargs': {'lr': 1e-3, 'weight_decay': 1e-5},
                    'lr_scheduler_cls': CosineAnnealingLR,
                    'lr_scheduler_kwargs': {'T_max': self.n_epochs, 'eta_min': 1e-4},
                    'model_name': self.emulator_name,
                },
                'automatic_params': {
                    'input_chunk_length': self.lookback,
                    'output_chunk_length': self.horizon
                },
                'supports_past_covariates': True,
                'supports_future_covariates': True,
                'historical_forecast_params': {'retrain': False},
                'supports_pl_trainer': True,
            },
            'TiDEModel': {
                    'model_class': 'TiDEModel',
                    'init_params': {
                        'batch_size': 128,
                        'optimizer_kwargs': {'lr': 1e-3, 'weight_decay': 1e-5},
                        'lr_scheduler_cls': CosineAnnealingLR,
                        'lr_scheduler_kwargs': {'T_max': self.n_epochs, 'eta_min': 1e-4},
                        'model_name': self.emulator_name,
                    },
                    'automatic_params': {
                        'input_chunk_length': self.lookback,
                        'output_chunk_length': self.horizon
                    },
                    'supports_past_covariates': True,
                    'supports_future_covariates': True,
                    'historical_forecast_params': {'retrain': False},
                    'supports_pl_trainer': True,
                },
        }

        # Initialize model class
        self.model_class = globals()[self.model_configs[self.model_handle]['model_class']]
        
        # Configure PyTorch Lightning trainer if needed
        self._configure_pl_trainer()

    def _configure_pl_trainer(self) -> None:
        """Configure PyTorch Lightning trainer if applicable."""
        if (self.pl_trainer_kwargs is None and 
            self.model_configs[self.model_handle]['supports_pl_trainer']):
            self.lr_logger = LearningRateMonitor()
            tb_logger = TensorBoardLogger(
                save_dir=self.log_path,
                name=self.emulator_name
            )
            
            self.pl_trainer_kwargs = {
                'accelerator': 'gpu',
                'devices': [self.gpu_id],
                'max_epochs': self.n_epochs,
                'callbacks': [self.lr_logger],
                'logger': tb_logger,
                'limit_train_batches': 1.0,
            }

            if self.try_early_stop:
                early_stopper = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    min_delta=1e-4,
                    mode='min',
                )
                self.pl_trainer_kwargs['callbacks'].append(early_stopper)

    def fit(
        self,
        use_all_data: bool = False,
        use_scaled_data: bool = True,
        save_after_fit: bool = False
    ) -> None:
        """Train the model on the provided data.
        
        Args:
            use_all_data: Whether to use entire dataset for training without validation split
            use_scaled_data: Whether to use scaled data for training
            save_after_fit: Whether to save the model after training
        """
        # Select appropriate data based on scaling preference
        targets = self.dev_dts.piped_targets if use_scaled_data else self.dev_dts.targets
        past_covariates = (self.dev_dts.piped_past_covariates 
                        if use_scaled_data else self.dev_dts.past_covariates)
        future_covariates = (self.dev_dts.piped_future_covariates 
                            if use_scaled_data else self.dev_dts.future_covariates)

        # Force using all data for certain models
        if self.model_class in [NaiveMean, NaiveSeasonal, NaiveMovingAverage,
                            NaiveDrift, ARIMA, AutoARIMA, LinearRegressionModel]:
            use_all_data = True

        # Prepare training data
        fit_kwargs = self._prepare_fit_kwargs(
            use_all_data, targets, past_covariates, future_covariates
        )

        # Add additional training parameters
        self._add_training_params(fit_kwargs)

        # Fit the model
        self._fit_model(fit_kwargs, use_all_data)

        if save_after_fit:
            self.save()

    def _prepare_fit_kwargs(
        self,
        use_all_data: bool,
        targets: TimeSeries,
        past_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries]
    ) -> Dict[str, Any]:
        """Prepare kwargs for model fitting."""
        if use_all_data:
            fit_kwargs = {'series': targets}
        else:
            # Split data into train and validation sets
            self.train_series, self.val_series = seasonal_split(
                targets, self.n_chunks, self.val_ratio
            )
            fit_kwargs = {
                'series': self.train_series,
                'val_series': self.val_series,
            }

            # Handle covariates
            if self.config['supports_past_covariates'] and past_covariates is not None:
                self.train_past_covariates, self.val_past_covariates = seasonal_split(
                    past_covariates, self.n_chunks, self.val_ratio
                )
                fit_kwargs.update({
                    'past_covariates': self.train_past_covariates,
                    'val_past_covariates': self.val_past_covariates
                })

            if self.config['supports_future_covariates'] and future_covariates is not None:
                self.train_future_covariates, self.val_future_covariates = seasonal_split(
                    future_covariates, self.n_chunks, self.val_ratio
                )
                fit_kwargs.update({
                    'future_covariates': self.train_future_covariates,
                    'val_future_covariates': self.val_future_covariates
                })

        return fit_kwargs

    def _add_training_params(self, fit_kwargs: Dict[str, Any]) -> None:
        """Add additional training parameters to fit_kwargs."""
        if hasattr(self.model, 'n_epochs'):
            fit_kwargs['epochs'] = self.n_epochs

        if self.pl_trainer_kwargs:
            fit_kwargs['trainer'] = Trainer(**self.pl_trainer_kwargs)

        if self.max_samples_per_ts is not None:
            fit_kwargs['max_samples_per_ts'] = self.max_samples_per_ts

        if self.dataloader_kwargs:
            fit_kwargs['dataloader_kwargs'] = self.dataloader_kwargs

        if 'verbose' in inspect.signature(self.model.fit).parameters:
            fit_kwargs['verbose'] = True

    def _fit_model(self, fit_kwargs: Dict[str, Any], use_all_data: bool) -> None:
        """Fit the model with the prepared parameters."""
        data_chunks = 1 if use_all_data else len(self.train_series)
        if not use_all_data and data_chunks > 1:
            print(f"Training on {data_chunks} chunks.")
        self.model.fit(**fit_kwargs)

    def save(self, path: Optional[str] = None) -> None:
        """Save the trained model to disk.
        
        Args:
            path: Path to save the model. If None, uses default path.
        """
        path = path or self.trained_model_save_path
        self.model.save(path)

    def load(self, path: Optional[str] = None) -> None:
        """Load a trained model from disk.
        
        Args:
            path: Path to load the model from. If None, uses default path.
        
        Raises:
            FileNotFoundError: If the model file is not found.
            IOError: If there's an error reading the model file.
        """
        path = path or self.trained_model_save_path
        try:
            self.model = self.model.load(path)
        except (FileNotFoundError, IOError) as e:
            print(f"Error loading the model: {e}")
            raise

    def historical_forecasts(
        self,
        test_dts: Optional[Any] = None,  # DartsTimeSeries type
        forecast_horizon: Optional[int] = None,
        original_scale: bool = True,
        last_points_only: bool = True,
        **kwargs: Any
    ) -> List[TimeSeries]:
        """Generate historical forecasts using the trained model.
        
        Args:
            test_dts: Test dataset (optional, overrides existing test data)
            forecast_horizon: Number of steps to forecast (optional)
            original_scale: Whether to return predictions in original scale
            last_points_only: Whether to return only the last points
            **kwargs: Additional arguments passed to the model's predict method
        
        Returns:
            List of TimeSeries objects containing the predictions
        """
        # Handle test data
        if test_dts is not None:
            print('Warning: Using the new test_dts to override the existing test_dts.')
            self.test_dts = test_dts
        elif self.test_dts is None:
            print("Warning: No test data was provided. Using the training data.")
            self.test_dts = self.dev_dts

        if forecast_horizon is not None:
            self.test_horizon = forecast_horizon

        # Prepare prediction kwargs
        predict_kwargs = {
            "series": self.test_dts.piped_targets,
            "forecast_horizon": self.test_horizon,
            "retrain": self.config['historical_forecast_params']['retrain'],
            "last_points_only": last_points_only
        }

        # Add covariates if supported
        if self.config.get('supports_past_covariates', False):
            predict_kwargs["past_covariates"] = self.test_dts.piped_past_covariates

        if self.config.get('supports_future_covariates', False):
            predict_kwargs["future_covariates"] = self.test_dts.piped_future_covariates

        predict_kwargs.update(kwargs)
        
        # Generate predictions
        self.hist_preds = self.model.historical_forecasts(**predict_kwargs)

        # Scale back predictions if requested
        if original_scale:
            self.hist_preds = [
                self.test_dts.target_pipeline.inverse_transform(ts)
                for ts in self.hist_preds
            ]

        return self.hist_preds

    def __str__(self) -> str:
        """Return string representation of the model."""
        return f"Model based on {self.model.__class__.__name__}"