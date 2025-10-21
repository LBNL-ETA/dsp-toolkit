from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from scipy import optimize, stats
from matplotlib import pyplot as plt
from tqdm import tqdm
from math import isclose

# Default thresholds for model validation
DEFAULT_R2_THRESHOLD = 0.7
DEFAULT_CVRMSE_THRESHOLD = 0.3
DEFAULT_SIGNIFICANT_PVAL = 0.05

class ChangePointModel:
    """
    Creates and fits a change-point model for building data analysis.
    
    A change-point model identifies relationships between a dependent variable 
    (typically energy use) and temperature, detecting points where the relationship 
    changes significantly. The model can identify different operating modes:
    - 1P (constant/base load)
    - 3P cooling (base + cooling)
    - 3P heating (heating + base)
    - 5P (heating + base + cooling)
    """

    def __init__(
        self,
        temperature: Union[np.ndarray, List[float]],
        y: Union[np.ndarray, List[float]],
        y_var_name: str = "Energy",
        min_r_squared: float = DEFAULT_R2_THRESHOLD,
        max_cv_rmse: float = DEFAULT_CVRMSE_THRESHOLD,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the ChangePointModel instance.

        Args:
            temperature: Array of temperature data points
            y: Array of dependent variable data (e.g., energy use, water usage)
            y_var_name: Name of the dependent variable for plotting and reporting
            min_r_squared: Minimum acceptable R² value for a valid model
            max_cv_rmse: Maximum acceptable CV(RMSE) value for a valid model
            verbose: If True, shows progress bar during model fitting

        Raises:
            ValueError: If input arrays are empty, different lengths, or contain invalid data
        """
        # Input validation
        if not np.any(temperature):
            raise ValueError("Temperature array must have at least one element")
        if not np.any(y):
            raise ValueError("Dependent variable array must have at least one element")
        if np.size(y) != np.size(temperature):
            raise ValueError("Temperature and dependent variable arrays must have the same length")
        if np.all(np.isnan(temperature)):
            raise ValueError("Temperature data contains only NaN values")

        # Store instance variables
        self.temperature = np.array(temperature)
        self.y = np.array(y)
        self.y_var_name = y_var_name
        self.min_r_squared = min_r_squared
        self.max_cv_rmse = max_cv_rmse
        self.verbose = verbose

        # Initialize model bounds
        self._initialize_bounds()

        # Initialize results storage
        self.ls_fit_results = None
        self.opt_fit = None
        self.valid_fit = False

    def _initialize_bounds(self) -> None:
        """Initialize the bounds for model parameters."""
        self.bounds = [
            # [hsl_min, hcp_min, base_min, ccp_min, csl_min]
            [-np.inf, np.min(self.temperature), np.min(self.y), 
             np.min(self.temperature), 0],
            # [hsl_max, hcp_max, base_max, ccp_max, csl_max]
            [0, np.max(self.temperature), np.max(self.y),
             np.max(self.temperature), np.inf]
        ]

    @staticmethod
    def piecewise_linear(x: np.ndarray, hsl: float, hcp: float, base: float, 
                        ccp: float, csl: float) -> np.ndarray:
        """
        Compute piecewise linear function values.
        
        Args:
            x: Input temperature values
            hsl: Heating slope (for temperatures < hcp)
            hcp: Heating change point temperature
            base: Base load value
            ccp: Cooling change point temperature
            csl: Cooling slope (for temperatures > ccp)
            
        Returns:
            Array of predicted values
        """
        if base is None:
            return np.nan
        
        # Handle 1P models (only baseload)
        if hcp is None and hsl is None and ccp is None and csl is None:
            return np.repeat(base, len(x))

        # Handle 3P models
        if (hcp is None and hsl is None) or (np.isnan(hcp) and np.isnan(hsl)):
            hcp = ccp
            hsl = 0
        if (ccp is None and csl is None) or (np.isnan(csl) and np.isnan(csl)):
            ccp = hcp
            csl = 0

        # Define piecewise conditions and functions
        conds = [x < hcp, (x >= hcp) & (x <= ccp), x > ccp]
        funcs = [
            lambda x: hsl * x + base - hsl * hcp,  # Heating region
            lambda x: base,                        # Base load region
            lambda x: csl * x + base - csl * ccp   # Cooling region
        ]

        return np.piecewise(x, conds, funcs)

    @staticmethod
    def calculate_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
        """Calculate the R² value for model predictions."""
        residuals = y - y_hat
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def calculate_cvrmse(y: np.ndarray, y_hat: np.ndarray) -> float:
        """Calculate the Coefficient of Variation of Root Mean Square Error."""
        return np.sqrt(np.sum((y - y_hat) ** 2) / len(y)) / np.mean(y)

    def fit_model(self) -> bool:
        """
        Fit the change-point model to the data.
        
        Returns:
            bool: True if a valid model was found, False otherwise
        """
        # Create search bounds for change points
        ls_cp_bounds = self._make_cp_bounds(n_bins=8)
        ls_fit_results = []

        # Try different change point bounds
        iterable = tqdm(ls_cp_bounds, desc=f"Fitting {self.y_var_name} model") if self.verbose else ls_cp_bounds
        
        for cp_bounds in iterable:
            try:
                bounds = self._update_cp_bounds(cp_bounds)
                fit_result = self._fit_once(bounds)
                ls_fit_results.append(fit_result)
            except Exception:
                continue

        self.ls_fit_results = ls_fit_results
        
        # Find the best model
        self.opt_fit = self._select_best_model()
        self.valid_fit = self.opt_fit['valid_fit']
        
        return self.valid_fit

    def predict(self, temperature: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            temperature: Optional temperature values for prediction. If None,
                        uses the training temperatures
                        
        Returns:
            Array of predicted values
        
        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if not self.opt_fit:
            raise ValueError("Model must be fitted before making predictions")
            
        temp = temperature if temperature is not None else self.temperature
        popt = [self.opt_fit[param] for param in ['hsl', 'hcp', 'base', 'ccp', 'csl']]
        return self.piecewise_linear(temp, *popt)

    def plot_model(self, title_prefix: Optional[str] = None, 
                  save_path: Optional[str] = None,
                  figsize: Tuple[int, int] = (10, 4)) -> Optional[Tuple]:
        """
        Plot the change-point model fit.
        
        Args:
            title_prefix: Optional prefix for the plot title
            save_path: Optional path to save the plot
            figsize: Figure size as (width, height)
            
        Returns:
            Tuple of (figure, axis) if plotting succeeds, None otherwise
        """
        if not self.opt_fit or not self.valid_fit:
            return self._plot_raw_data(title_prefix, figsize)
            
        return self._plot_fitted_model(title_prefix, save_path, figsize)

    @classmethod
    def no_fit_output(cls) -> Dict:
        """Return a dictionary of null results for when no valid fit is found."""
        return {
            'hsl': None, 'hcp': None, 'base': None, 'ccp': None, 'csl': None,
            'r2': None, 'cvrmse': None, 'pval_hsl': None, 'pval_csl': None,
            'valid_hsl': False, 'valid_csl': False, 'model type': None,
            'valid_fit': False
        }

    def _make_cp_bounds(self, n_bins: int = 4) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Create bounds for heating and cooling change-points search.
        
        Args:
            n_bins: Number of temperature bins to use for search
            
        Returns:
            List of tuples containing (heating_cp_bounds, cooling_cp_bounds)
        """
        data = self.temperature
        bin_width = np.ptp(data) / n_bins
        marks = [min(data) + i * bin_width for i in range(n_bins + 1)]
        
        bounds = []
        for i in range(len(marks) - 1):
            for j in range(i + 1, len(marks) - 1):
                bounds.append([
                    (marks[i], marks[i + 1]),     # heating cp bounds
                    (marks[j], marks[j + 1])      # cooling cp bounds
                ])
        return bounds

    def _update_cp_bounds(self, cp_bounds: Tuple[Tuple[float, float], Tuple[float, float]]) -> List[List[float]]:
        """Update the model bounds with new change point bounds."""
        bounds = [self.bounds[0].copy(), self.bounds[1].copy()]
        
        # Update heating change point bounds
        bounds[0][1] = cp_bounds[0][0]  # hcp lower
        bounds[1][1] = cp_bounds[0][1]  # hcp upper
        
        # Update cooling change point bounds
        bounds[0][3] = cp_bounds[1][0]  # ccp lower
        bounds[1][3] = cp_bounds[1][1]  # ccp upper
        
        return bounds

    def _fit_once(self, bounds: List[List[float]]) -> Dict:
        """
        Attempt one model fit with given bounds.
        
        Args:
            bounds: List of [lower_bounds, upper_bounds] for all parameters
            
        Returns:
            Dictionary containing fit results and statistics
        """
        # Perform curve fit
        popt, pcov = optimize.curve_fit(
            f=self.piecewise_linear,
            xdata=self.temperature,
            ydata=self.y,
            bounds=bounds,
            method='dogbox'
        )
        
        # Calculate model quality metrics
        y_pred = self.piecewise_linear(self.temperature, *popt)
        r2 = self.calculate_r2(self.y, y_pred)
        cvrmse = self.calculate_cvrmse(self.y, y_pred)
        
        # Check significance of slopes
        pval_hsl = pval_csl = None
        valid_hsl = valid_csl = False
        
        # Check heating slope significance
        if not isclose(popt[0], 0, abs_tol=1e-5):
            x_heat = self.temperature[self.temperature <= popt[1]]
            y_heat = self.y[self.temperature <= popt[1]]
            y_heat_pred = self.piecewise_linear(x_heat, *popt)
            pval_hsl = self._calculate_slope_pval(popt[0], x_heat, y_heat, y_heat_pred)
            valid_hsl = pval_hsl < DEFAULT_SIGNIFICANT_PVAL
            
        # Check cooling slope significance
        if not isclose(popt[4], 0, abs_tol=1e-5):
            x_cool = self.temperature[self.temperature >= popt[3]]
            y_cool = self.y[self.temperature >= popt[3]]
            y_cool_pred = self.piecewise_linear(x_cool, *popt)
            pval_csl = self._calculate_slope_pval(popt[4], x_cool, y_cool, y_cool_pred)
            valid_csl = pval_csl < DEFAULT_SIGNIFICANT_PVAL
            
        return {
            "popt": popt,
            "pcov": pcov,
            "r2": r2,
            "cvrmse": cvrmse,
            "pval_hsl": pval_hsl,
            "pval_csl": pval_csl,
            "valid_hsl": valid_hsl,
            "valid_csl": valid_csl
        }

    def _calculate_slope_pval(self, slope: float, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the p-value for a regression slope.
        
        Args:
            slope: Slope value to test
            x: Input temperature values
            y: Actual dependent variable values
            y_pred: Predicted dependent variable values
            
        Returns:
            P-value for the slope
        """
        if len(x) <= 2:
            return np.inf
            
        # Calculate sample variance and standard error
        sample_variance = np.sum((y - y_pred) ** 2) / (len(x) - 2)
        sum_square_x = np.sum((x - np.mean(x)) ** 2)
        std_error = np.sqrt(sample_variance / sum_square_x)
        
        # Calculate t-statistic and p-value
        t_score = slope / std_error
        pval = stats.t.sf(np.abs(t_score), len(x) - 1) * 2  # two-tailed test
        
        return pval

    def _select_best_model(self) -> Dict:
        """
        Select the best model from all fitted results.
        
        Returns:
            Dictionary containing the best model parameters and metadata
        """
        if not self.ls_fit_results:
            return self.no_fit_output()
            
        # Convert results to DataFrame for easier analysis
        rows = []
        for fr in self.ls_fit_results:
            row = fr['popt'].tolist() + [
                fr['r2'], fr['cvrmse'], fr['pval_hsl'], 
                fr['pval_csl'], fr['valid_hsl'], fr['valid_csl']
            ]
            rows.append(row)
            
        df_fits = pd.DataFrame(
            rows,
            columns=['hsl', 'hcp', 'base', 'ccp', 'csl', 'r2', 'cvrmse',
                    'pval_hsl', 'pval_csl', 'valid_hsl', 'valid_csl']
        )
        
        # Filter for valid models (at least one valid slope)
        df_valid = df_fits.loc[(df_fits['valid_hsl']) | (df_fits['valid_csl'])]
        
        if len(df_valid) == 0:
            # Try 1P model if no valid 3P or 5P models
            opt_fit = {
                'hsl': None, 'hcp': None,
                'base': np.mean(self.y),
                'ccp': None, 'csl': None,
                'r2': self.calculate_r2(self.y, np.mean(self.y)),
                'cvrmse': self.calculate_cvrmse(self.y, np.mean(self.y)),
                'valid_hsl': False, 'valid_csl': False,
                'pval_hsl': None, 'pval_csl': None
            }
            
            if opt_fit['cvrmse'] <= self.max_cv_rmse:
                opt_fit.update({'valid_fit': True, 'model type': '1P'})
            else:
                opt_fit.update({'valid_fit': False, 'model type': 'No-fit'})
                
            return opt_fit
        
        # Select model with highest R² from valid models
        opt_fit = df_valid.loc[df_valid['r2'].idxmax()].to_dict()
        opt_fit['valid_fit'] = False
        
        # Determine model type based on valid slopes
        if opt_fit['valid_hsl'] and opt_fit['valid_csl']:
            # 5P model (both heating and cooling)
            opt_fit['model type'] = '5P'
            opt_fit['valid_fit'] = True
            
        elif opt_fit['valid_hsl']:
            # 3P heating model
            opt_fit['model type'] = '3P Heating'
            opt_fit['ccp'] = None
            opt_fit['csl'] = None
            opt_fit['valid_fit'] = True
            
        elif opt_fit['valid_csl']:
            # 3P cooling model
            opt_fit['model type'] = '3P Cooling'
            opt_fit['hsl'] = None
            opt_fit['hcp'] = None
            opt_fit['valid_fit'] = True
            
        return opt_fit

    def _plot_raw_data(self, title_prefix: Optional[str], figsize: Tuple[int, int]) -> Tuple:
        """Plot raw data when no valid model is found."""
        f, ax = plt.subplots(1, 1, figsize=figsize)
        ax.scatter(self.temperature, self.y, alpha=0.5, s=10, c='k')
        ax.set_ylim(0, self.y.max() * 1.1)
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel(self.y_var_name)
        
        # Add border lines to the plots
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        
        title = f"Raw Data"
        if title_prefix:
            title = f"{title} {title_prefix}"
        title = f"{title}: Temperature vs {self.y_var_name} ({len(self.y)} points)"
        if not self.opt_fit:
            title += "; no model fitted yet"
        else:
            title += "; no valid model fit"
            
        ax.set_title(title)
        f.tight_layout()
        return f, ax

    def _plot_fitted_model(self, title_prefix: Optional[str], save_path: Optional[str], 
                          figsize: Tuple[int, int]) -> Tuple:
        """Plot the fitted model with data."""
        f, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot data points
        ax.scatter(self.temperature, self.y, alpha=0.5, s=10, c='k')
        
        # Generate prediction points
        padding = 5 * (self.temperature.max() - self.temperature.min()) / 100
        x_range = [self.temperature.min() - padding, self.temperature.max() + padding]
        
        model_info = self._get_model_plot_info(x_range)
        
        # Plot model segments
        for segment in model_info['segments']:
            ax.plot(segment['x'], segment['y'], c=segment['color'], linewidth=2)
            
        # Set plot properties
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        
        ax.set_ylim(0, self.y.max() * 1.1)
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel(self.y_var_name)
        ax.set_title(f"Change-point model for {self.y_var_name} ({len(self.y)} points)")
        
        # Add model information text
        ax.text(1.02, 0.95, model_info['text'], transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        f.tight_layout()
        if save_path:
            f.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return f, ax

    def _get_model_plot_info(self, x_range: List[float]) -> Dict:
        """
        Prepare model information for plotting.
        
        Args:
            x_range: List containing [min_x, max_x] for plotting range
            
        Returns:
            Dictionary containing:
                - segments: List of dictionaries with x, y coordinates and colors for each model segment
                - text: String containing model information for the plot annotation
        """
        opt_fit = self.opt_fit
        model_type = opt_fit['model type']
        
        info = {
            'segments': [],
            'text': f"Model: {model_type}\nR²: {opt_fit['r2']:.2f}\n"
        }
        
        if model_type == '1P':
            # Constant model - single horizontal line
            x = np.linspace(*x_range, 200)
            info['segments'].append({
                'x': x,
                'y': np.full_like(x, opt_fit['base']),
                'color': 'k'
            })
            info['text'] += f"Base: {opt_fit['base']:.2f}\n"
            info['text'] += f"CV(RMSE): {opt_fit['cvrmse']:.2f}"
            
        elif model_type == '3P Heating':
            # 3P Heating model - heating slope and base segments
            x_heat = np.linspace(x_range[0], opt_fit['hcp'], 100)
            x_base = np.linspace(opt_fit['hcp'], x_range[1], 100)
            info['segments'].extend([
                {
                    'x': x_heat,
                    'y': self.predict(x_heat),
                    'color': 'r'
                },
                {
                    'x': x_base,
                    'y': self.predict(x_base),
                    'color': 'k'
                }
            ])
            info['text'] += (
                f"Heating Slope: {opt_fit['hsl']:.2f}\n"
                f"Heating Change-point: {opt_fit['hcp']:.2f}\n"
                f"Base: {opt_fit['base']:.2f}"
            )
            
        elif model_type == '3P Cooling':
            # 3P Cooling model - base and cooling slope segments
            x_base = np.linspace(x_range[0], opt_fit['ccp'], 100)
            x_cool = np.linspace(opt_fit['ccp'], x_range[1], 100)
            info['segments'].extend([
                {
                    'x': x_base,
                    'y': self.predict(x_base),
                    'color': 'k'
                },
                {
                    'x': x_cool,
                    'y': self.predict(x_cool),
                    'color': 'b'
                }
            ])
            info['text'] += (
                f"Base: {opt_fit['base']:.2f}\n"
                f"Cooling Change-point: {opt_fit['ccp']:.2f}\n"
                f"Cooling Slope: {opt_fit['csl']:.2f}"
            )
            
        elif model_type == '5P':
            # 5P model - heating, base, and cooling segments
            x_heat = np.linspace(x_range[0], opt_fit['hcp'], 100)
            x_base = np.linspace(opt_fit['hcp'], opt_fit['ccp'], 100)
            x_cool = np.linspace(opt_fit['ccp'], x_range[1], 100)
            info['segments'].extend([
                {
                    'x': x_heat,
                    'y': self.predict(x_heat),
                    'color': 'r'
                },
                {
                    'x': x_base,
                    'y': self.predict(x_base),
                    'color': 'k'
                },
                {
                    'x': x_cool,
                    'y': self.predict(x_cool),
                    'color': 'b'
                }
            ])
            info['text'] += (
                f"Heating Slope: {opt_fit['hsl']:.2f}\n"
                f"Heating Change-point: {opt_fit['hcp']:.2f}\n"
                f"Base: {opt_fit['base']:.2f}\n"
                f"Cooling Change-point: {opt_fit['ccp']:.2f}\n"
                f"Cooling Slope: {opt_fit['csl']:.2f}"
            )
        
        return info




