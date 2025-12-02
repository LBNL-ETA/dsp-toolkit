# Distribution Substation Planning Toolkit

## Overview
This repository is a comprehensive Python toolkit designed for **Distribution Substation Planning (DSP)** in electric power systems. The toolkit addresses three critical aspects of electric grid planning: data curation, short-term load forecasting, and weather-sensitive load modeling for future peak demand predictions.

The toolkit enables utility engineers and power system analysts to predict electrical load patterns at distribution substations. This is essential for infrastructure planning, ensuring substations can handle future demand without overinvestment in capacity. The software combines modern deep learning techniques with traditional statistical methods to provide robust forecasting capabilities.

## Key Features

- **Comprehensive Forecasting Models**: Support for 15+ forecasting methods ranging from naive baselines to state-of-the-art deep learning models
- **Weather Sensitivity Analysis**: Automated change-point detection for temperature-dependent load patterns
- **Data Pipeline Management**: Built-in tools for time series preprocessing, scaling, and covariate handling
- **GPU Acceleration**: PyTorch Lightning integration for efficient deep learning model training
- **Model Validation**: Automatic performance metrics including R², CV(RMSE), and statistical significance testing
- **Production Ready**: Model persistence, TensorBoard logging, and comprehensive error handling

## Core Modules

### 1. Short-Term Forecasting Models
The `Model` class in [short_term_forecasting_models.py](src/ds_planner/short_term_forecasting_models.py) provides a unified interface for multiple forecasting approaches:

**Statistical Methods:**
- Naive baselines (Mean, Seasonal, Moving Average, Drift)
- ARIMA and AutoARIMA

**Machine Learning Methods:**
- Linear Regression (multi-model and autoregressive)
- Random Forest
- XGBoost
- LightGBM

**Deep Learning Methods:**
- RNN/LSTM (RNNModel, BlockRNNModel)
- Temporal Convolutional Networks (TCNModel)
- Temporal Fusion Transformer (TFTModel)
- N-BEATS and N-HiTS
- TSMixer
- TiDE (Time-series Dense Encoder)

Features include automated hyperparameter management, support for past and future covariates (e.g., weather data), configurable lookback windows and forecast horizons, and early stopping with learning rate scheduling.

### 2. Weather Sensitivity Analysis
The `ChangePointModel` class in [weather_sensitivity_models.py](src/ds_planner/weather_sensitivity_models.py) implements piecewise linear regression to identify temperature-dependent load patterns:

**Model Types:**
- **1P**: Constant base load (no temperature sensitivity)
- **3P Heating**: Heating-sensitive load with change point
- **3P Cooling**: Cooling-sensitive load with change point
- **5P**: Both heating and cooling sensitivity with dual change points

The model automatically detects change points where energy consumption behavior shifts (typically corresponding to heating/cooling thresholds) and validates results using R² thresholds (default 0.7), CV(RMSE) limits (default 0.3), and statistical significance testing.

### 3. Time Series Data Management
The `DartsTimeSeries` wrapper in [time_series.py](src/ds_planner/time_series.py) provides:
- Automated data preprocessing and validation
- Support for target variables (loads) and covariates (weather)
- Built-in scaling pipelines
- Seasonal data splitting for training/validation

## Installation

### Step 0 (optional, recommended): Use a Virtual Environment

It is recommended to use a virtual environment to manage dependencies. You can create a new virtual environment using `venv` or `conda`.

### Step 1: Install PyTorch

Building Data Copilot depends on PyTorch. Please install the appropriate version of PyTorch based on your system and CUDA setup. Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

Example for CUDA 11.8:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only installation:
```
pip install torch
```

### Step 2: Install the `dsp-toolkit`

Once PyTorch is installed, you can install the toolkit:
```
cd dsp_toolkit
pip install .
```

For development mode, you can install the toolkit in editable mode, along with dev tools:
```
cd dsp_toolkit
pip install -e .[dev]
```


## Usage

### Quick Start Example

```python
import pandas as pd
from ds_planner.short_term_forecasting_models import Model
from ds_planner.time_series import DartsTimeSeries
from ds_planner.data import load_dataset

# Load example data
df = load_dataset('example_hourly_data')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

# Create time series object
target_cols = ['Electricity_kWh', 'NaturalGas_kWh']
past_cov_cols = ['T_out', 'RH_out', 'Wind_speed_m_per_s',
                 'Wind_direction_deg', 'Diffuse_Solar_W_per_m2',
                 'Direct_Solar_W_per_m2']
dts = DartsTimeSeries(df, freq='h', target_cols=target_cols,
                      past_cov_cols=past_cov_cols)

# Train a forecasting model
model = Model(
    model_handle="TFTModel",
    dev_dts=dts,
    lookback=24,      # Use 24 hours of history
    horizon=6,        # Forecast 6 hours ahead
    n_epochs=50
)
model.fit()
model.save()

# Generate predictions
predictions = model.historical_forecasts()
```

### Weather Sensitivity Analysis

```python
from ds_planner.weather_sensitivity_models import ChangePointModel

# Resample to monthly data for better change-point detection
df_monthly = df.resample('ME').agg({
    'T_out': 'mean',
    'Electricity_kWh': 'sum'
})

# Fit change-point model
cp_model = ChangePointModel(
    temperature=df_monthly['T_out'],
    y=df_monthly['Electricity_kWh'],
    y_var_name='Electricity (kWh)'
)
cp_model.fit_model()

# Plot results
cp_model.plot_model()

# Make predictions for future temperatures
future_temps = [10, 15, 20, 25, 30]
predictions = cp_model.predict(temperature=future_temps)
```

### Complete Examples
For comprehensive examples including data loading, model comparison, and visualization, see the [example notebook](notebooks/example.ipynb).

## Use Cases

This toolkit is designed for power system engineers, utility planners, and energy researchers working on:

- **Substation Capacity Planning**: Predict future peak demand to determine when upgrades are needed
- **Electrification Planning**: Forecast load growth from electric vehicles and heat pump adoption
- **Renewable Integration**: Model load patterns to support distributed energy resource planning
- **Climate Adaptation**: Assess how changing temperature patterns affect electricity and gas demand
- **Operational Planning**: Short-term forecasts for day-ahead and real-time system operations

The toolkit is particularly relevant for organizations planning for grid modernization where accurate load forecasting becomes increasingly complex due to distributed generation, electrification, and climate change.

## Example Datasets

The toolkit includes example datasets demonstrating typical use cases:
- **example_hourly_data.csv**: Hourly electricity and natural gas consumption with weather variables
- **example_weather.csv**: Weather data including temperature, humidity, wind speed/direction, and solar radiation

These datasets can be accessed via:
```python
from ds_planner.data import get_available_datasets, load_dataset
print(get_available_datasets())
df = load_dataset('example_hourly_data')
```

## Dependencies

Built on a modern Python stack:
- **PyTorch**: Deep learning framework for neural network models
- **Darts**: Time series library providing the forecasting engine
- **Pandas & NumPy**: Data manipulation and numerical computing
- **PyTorch Lightning**: Training infrastructure with GPU support
- **Scipy & Matplotlib**: Statistical analysis and visualization

## Contributing
Contributions are welcome! If you'd like to report a bug, suggest a feature, or contribute code, please visit the GitHub repository and open an issue or pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or feedback, contact the author:
- **Han Li** - Lawrence Berkeley National Laboratory
- Email: hanli@lbl.gov

## Acknowledgments
This toolkit represents a modern approach to traditional utility planning challenges using cutting-edge machine learning techniques. It is designed to support the ongoing transformation of electric distribution systems as they integrate renewable energy, electrification, and climate adaptation strategies.
