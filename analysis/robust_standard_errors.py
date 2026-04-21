import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import check_X_y
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Configure module logger
logger = logging.getLogger(__name__)

# Check if dask is available
try:
    import dask.array as da
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

class RobustStandardErrors:
    """
    Calculate heteroskedasticity-consistent (HC) robust standard errors for regression models.
    
    This class implements HC-type robust standard errors, with a focus on HC2 which is 
    recommended for most economic and scientific applications. Supports both standard 
    scikit-learn models and dask-ml models with large datasets.
    
    HC types available:
    - HC0: White's estimator (original)
    - HC1: HC0 with small sample correction 
    - HC2: HC0 with leverage correction (recommended default)
    - HC3: HC0 with alternative leverage correction (more conservative)
    
    Parameters
    ----------
    estimator : scikit-learn or dask-ml estimator
        A fitted regression estimator with coef_ and predict() attributes.
    X : array-like or dask array
        The design matrix used for fitting.
    y : array-like or dask array
        The target values used for fitting.
    cov_type : str, default='HC2'
        The type of robust standard errors to compute.
        Options include: 'HC0', 'HC1', 'HC2', 'HC3'.
        
    Attributes
    ----------
    coef_ : ndarray
        The regression coefficients, including the intercept if applicable.
    std_errors_ : ndarray
        The robust standard errors.
    t_stats_ : ndarray
        The t-statistics.
    p_values_ : ndarray
        The p-values.
    conf_int_ : ndarray
        The 95% confidence intervals.
    cov_matrix_ : ndarray
        The covariance matrix of the coefficients.
    
    Notes
    -----
    HC2 is generally recommended as it corrects for both heteroskedasticity and leverage.
    For large datasets, the implementation will automatically use a more efficient approach
    when dask arrays are detected.
    
    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    >>> reg = LinearRegression().fit(X, y)
    >>> rse = RobustStandardErrors(reg, X, y).fit()
    >>> print(rse.summary())
    
    With dask-ml:
    >>> import dask.array as da
    >>> from dask_ml.linear_model import LinearRegression
    >>> X = da.random.normal(size=(10000, 10), chunks=(1000, 10))
    >>> y = da.random.normal(size=(10000,), chunks=(1000,))
    >>> reg = LinearRegression().fit(X, y)
    >>> rse = RobustStandardErrors(reg, X, y).fit()
    >>> print(rse.summary())
    """
    
    def __init__(self, estimator, X, y, cov_type='HC2'):
        self.estimator = estimator
        self.X_orig = X
        self.cov_type = cov_type
        
        # Check if estimator is fitted
        if not hasattr(estimator, 'coef_'):
            raise ValueError("Estimator is not fitted yet. Call 'fit' before using this method.")
        
        # Check valid cov_type
        valid_cov_types = ['HC0', 'HC1', 'HC2', 'HC3']
        if cov_type not in valid_cov_types:
            raise ValueError(f"cov_type must be one of {valid_cov_types}")
        
        # Prepare design matrix and target
        self._prepare_data(X, y)
    
    @classmethod
    def from_sklearn(cls, model, X, y, cov_type='HC2'):
        """
        Create a RobustStandardErrors instance from a fitted scikit-learn model.
        
        Parameters
        ----------
        model : scikit-learn model
            A fitted scikit-learn regression model.
        X : array-like
            The design matrix used for fitting.
        y : array-like
            The target values used for fitting.
        cov_type : str, default='HC2'
            The type of robust standard errors to compute.
            
        Returns
        -------
        RobustStandardErrors
            A fitted RobustStandardErrors instance.
        """
        return cls(model, X, y, cov_type=cov_type).fit()
    
    def _prepare_data(self, X, y):
        """
        Prepare the design matrix and target.
        
        Parameters
        ----------
        X : array-like
            The design matrix.
        y : array-like
            The target vector.
        """
        # Check if X and y are valid, handle dask arrays specially
        if HAS_DASK and (isinstance(X, da.Array) or isinstance(y, da.Array)):
            # Using dask arrays - note but don't convert yet
            self.is_dask = True
            self.X_fit = X
            self.y = y
            logger.info("Dask arrays detected, will use optimized computation")
        else:
            # Standard numpy/pandas data
            self.is_dask = False
            try:
                X, y = check_X_y(X, y, y_numeric=True, multi_output=False)
                self.X_fit = X
                self.y = y
            except ValueError:
                # Handle the case where check_X_y fails (e.g. with dask arrays)
                self.X_fit = X
                self.y = y
        
        # Check if the model has intercept
        if hasattr(self.estimator, 'fit_intercept') and self.estimator.fit_intercept:
            self.has_intercept = True
        else:
            self.has_intercept = False
        
        # Get feature names
        self.feature_names = self._get_feature_names()
        
        # Calculate predictions and residuals
        self.y_pred = self.estimator.predict(X)
        
        # Handle dask arrays for residuals
        if self.is_dask:
            self.residuals = self.y - self.y_pred
        else:
            self.residuals = np.array(self.y - self.y_pred)
    
    def _get_feature_names(self):
        """
        Get feature names from the design matrix.
        
        Returns
        -------
        list
            The feature names, including the intercept if applicable.
        """
        if hasattr(self.X_orig, 'columns'):
            # It's a pandas DataFrame
            feature_names = list(self.X_orig.columns)
        else:
            # It's a numpy or dask array
            feature_names = [f'x{i}' for i in range(self.X_fit.shape[1])]
        
        # Add intercept if the model has one
        if self.has_intercept:
            feature_names = ['intercept'] + feature_names
        
        return feature_names
    
    def _build_design_matrix(self):
        """
        Build the design matrix with intercept if needed.
        
        Returns
        -------
        array-like
            The design matrix with intercept if needed.
        """
        if self.is_dask:
            # Handle dask arrays
            if self.has_intercept:
                # Add constant column for dask array
                ones = da.ones((self.X_fit.shape[0], 1))
                X = da.concatenate([ones, self.X_fit], axis=1)
                return X
            else:
                return self.X_fit
        else:
            # Handle numpy arrays
            if self.has_intercept:
                # Add constant column for numpy array
                ones = np.ones((self.X_fit.shape[0], 1))
                X = np.concatenate([ones, self.X_fit], axis=1)
                return X
            else:
                return self.X_fit
    
    def _get_coef(self):
        """
        Get coefficients, including intercept if the model has one.
        
        Returns
        -------
        ndarray
            The coefficients, including the intercept if applicable.
        """
        if self.has_intercept:
            if hasattr(self.estimator, 'intercept_'):
                # Standard scikit-learn models store intercept separately
                intercept = self.estimator.intercept_
                if np.isscalar(intercept):
                    intercept = np.array([intercept])
                return np.concatenate([intercept, self.estimator.coef_])
            else:
                # Some models might include intercept in coefficients
                return self.estimator.coef_
        else:
            return self.estimator.coef_
    
    def fit(self):
        """
        Calculate robust standard errors.
        
        Returns
        -------
        self : object
            Returns self.
        """
        # Build design matrix with intercept if needed
        self.X = self._build_design_matrix()
        
        # Get coefficients
        self.coef_ = self._get_coef()
        
        # For dask arrays, convert to numpy for robust calculations
        if self.is_dask:
            logger.info(f"Computing {self.cov_type} robust standard errors using numpy backend")
            self.cov_matrix_ = self._calculate_cov_matrix_with_numpy()
        else:
            # Use native numpy implementation
            logger.info(f"Computing {self.cov_type} robust standard errors")
            self.cov_matrix_ = self._calculate_cov_matrix_native()
        
        # Calculate standard errors
        self.std_errors_ = np.sqrt(np.diag(self.cov_matrix_))
        
        # Calculate t-statistics and p-values
        self.t_stats_ = self.coef_ / self.std_errors_
        
        # Degrees of freedom
        n = self.X.shape[0]
        k = self.X.shape[1]
        df = n - k
        
        self.p_values_ = 2 * (1 - stats.t.cdf(np.abs(self.t_stats_), df))
        
        # Calculate confidence intervals (95% by default)
        alpha = 0.05
        t_crit = stats.t.ppf(1 - alpha/2, df)
        self.conf_int_ = np.vstack([
            self.coef_ - t_crit * self.std_errors_,
            self.coef_ + t_crit * self.std_errors_
        ]).T
        
        return self
        
    def _calculate_cov_matrix_with_numpy(self):
        """
        Calculate covariance matrix using numpy for all HC types, converting from dask if needed.
        
        Returns
        -------
        ndarray
            The robust covariance matrix.
        """
        # Convert dask arrays to numpy arrays
        if self.is_dask:
            X_np = self.X.compute()
            residuals_np = self.residuals.compute()
        else:
            X_np = self.X
            residuals_np = self.residuals
        
        # Calculate X'X and its inverse
        XtX_np = X_np.T @ X_np
        XtX_inv_np = np.linalg.inv(XtX_np)
        
        # For HC0 and HC1, we don't need the hat matrix
        if self.cov_type in ['HC0', 'HC1']:
            # Calculate weights
            if self.cov_type == 'HC0':
                weights = residuals_np**2
            else:  # HC1
                n, k = X_np.shape
                weights = residuals_np**2 * (n / (n - k))
        else:
            # For HC2 and HC3, calculate hat matrix diagonal (leverage values)
            hat_diag_np = np.zeros(X_np.shape[0])
            for i in range(X_np.shape[0]):
                x_i = X_np[i:i+1]
                hat_diag_np[i] = (x_i @ XtX_inv_np @ x_i.T)[0, 0]
            
            # Handle potential division by zero (leverage points close to 1)
            max_leverage = 0.999  # Protect against division by zero
            hat_diag_np = np.clip(hat_diag_np, 0, max_leverage)
            
            # Calculate HC2 or HC3 weights
            if self.cov_type == 'HC2':
                weights = residuals_np**2 / (1 - hat_diag_np)
            else:  # HC3
                weights = residuals_np**2 / (1 - hat_diag_np)**2
        
        # Handle near-zero residuals
        eps = 1e-10
        weights[residuals_np**2 < eps] = eps
        
        # Calculate meat of sandwich (X' Ω X)
        meat = X_np.T @ (np.diag(weights) @ X_np)
        
        # Calculate the HC covariance matrix (X'X)^(-1) X' Ω X (X'X)^(-1)
        cov = XtX_inv_np @ meat @ XtX_inv_np
        return cov
    
    def _calculate_cov_matrix_native(self):
        """Calculate covariance matrix using native numpy operations."""
        X_np = self.X
        residuals_np = self.residuals
        
        # Calculate X'X and its inverse
        XtX_np = X_np.T @ X_np
        
        # Add small regularization term to handle multicollinearity
        epsilon = 1e-8
        XtX_np = XtX_np + epsilon * np.eye(XtX_np.shape[0])
        
        # Check condition number before attempting inversion
        cond_num = np.linalg.cond(XtX_np)
        if cond_num > 1e12:  # A very high threshold indicating severe multicollinearity
            raise ValueError(f"Matrix is singular or ill-conditioned (condition number: {cond_num:.2e}). "
                             f"This typically indicates perfect multicollinearity among predictors. "
                             f"Consider removing redundant variables or using regularization.")
        
        # Safely compute inverse (try pseudoinverse if regular inverse fails)
        try:
            XtX_inv_np = np.linalg.inv(XtX_np)
        except np.linalg.LinAlgError:
            print("    Warning: Matrix is singular, using pseudoinverse")
            XtX_inv_np = np.linalg.pinv(XtX_np)
        
        # For HC0 and HC1, we don't need the hat matrix
        if self.cov_type in ['HC0', 'HC1']:
            # Calculate weights
            if self.cov_type == 'HC0':
                weights = residuals_np**2
            else:  # HC1
                n, k = X_np.shape
                weights = residuals_np**2 * (n / (n - k))
        else:
            # For HC2 and HC3, calculate hat matrix diagonal (leverage values)
            hat_diag_np = np.zeros(X_np.shape[0])
            for i in range(X_np.shape[0]):
                x_i = X_np[i:i+1]
                hat_diag_np[i] = (x_i @ XtX_inv_np @ x_i.T)[0, 0]
            
            # Handle potential division by zero (leverage points close to 1)
            max_leverage = 0.999  # Protect against division by zero
            hat_diag_np = np.clip(hat_diag_np, 0, max_leverage)
            
            # Calculate HC2 or HC3 weights
            if self.cov_type == 'HC2':
                weights = residuals_np**2 / (1 - hat_diag_np)
            else:  # HC3
                weights = residuals_np**2 / (1 - hat_diag_np)**2
        
        # Handle near-zero residuals
        eps = 1e-10
        weights[residuals_np**2 < eps] = eps
        
        # Calculate meat of sandwich (X' Ω X)
        # REPLACE THIS LINE:
        # meat = X_np.T @ (np.diag(weights) @ X_np)
        
        # WITH THIS MEMORY-EFFICIENT VERSION:
        meat = X_np.T @ (weights[:, np.newaxis] * X_np)
        
        # Calculate the HC covariance matrix (X'X)^(-1) X' Ω X (X'X)^(-1)
        cov = XtX_inv_np @ meat @ XtX_inv_np
        return cov
    
    def summary(self):
        """
        Create a summary DataFrame with results.
        
        Returns
        -------
        summary : pandas.DataFrame
            A DataFrame containing the results.
        """
        # Create a summary DataFrame
        summary_df = pd.DataFrame({
            'Coefficient': self.coef_,
            f'Std Error ({self.cov_type})': self.std_errors_,
            't value': self.t_stats_,
            'P>|t|': self.p_values_,
            'Lower 95% CI': self.conf_int_[:, 0],
            'Upper 95% CI': self.conf_int_[:, 1]
        })
        
        # Add row names
        summary_df.index = self.feature_names
        
        return summary_df
    
#%% Class Tests

def validate_against_statsmodels(X, y, cov_type='HC2'):
    """
    Validate our implementation against statsmodels.
    
    Parameters
    ----------
    X : array-like
        The design matrix.
    y : array-like
        The target vector.
    cov_type : str, default='HC2'
        The type of robust standard errors to compute.
        
    Returns
    -------
    dict
        A dictionary with comparison results.
    """
    # Import statsmodels
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels is required for validation.")
    
    # Fit models
    # Scikit-learn model
    sklearn_model = LinearRegression(fit_intercept=True).fit(X, y)
    rse = RobustStandardErrors(sklearn_model, X, y, cov_type=cov_type).fit()
    
    # Statsmodels model (add constant for intercept)
    X_sm = sm.add_constant(X)
    sm_model = sm.OLS(y, X_sm).fit(cov_type=cov_type.lower())
    
    # Compare results
    comparison = {
        'sklearn_coef': rse.coef_,
        'statsmodels_coef': sm_model.params,
        'sklearn_se': rse.std_errors_,
        'statsmodels_se': sm_model.bse,
        'coef_difference': rse.coef_ - sm_model.params,
        'se_difference': rse.std_errors_ - sm_model.bse,
        'rel_coef_diff': (rse.coef_ - sm_model.params) / np.where(sm_model.params != 0, sm_model.params, 1),
        'rel_se_diff': (rse.std_errors_ - sm_model.bse) / sm_model.bse
    }
    
    return comparison

def test_robust_standard_errors():
    """Test the RobustStandardErrors implementation."""
    print("Testing RobustStandardErrors against statsmodels...")
    
    # Generate heteroskedastic data
    np.random.seed(42)
    n = 100
    k = 5
    X = np.random.normal(0, 1, (n, k))
    beta = np.array([0.5, 0.2, -0.3, 0.1, -0.2])
    
    # Create heteroskedastic errors (variance increases with X_0)
    e = np.random.normal(0, 1, n) * np.exp(X[:, 0])
    y = X @ beta + e
    
    # Fit a model using sklearn
    model = LinearRegression(fit_intercept=True).fit(X, y)
    
    # Calculate robust standard errors
    rse = RobustStandardErrors(model, X, y, cov_type='HC2').fit()
    
    try:
        # Import statsmodels
        import statsmodels.api as sm
        
        # Fit statsmodels model with same data
        X_sm = sm.add_constant(X)
        sm_model = sm.OLS(y, X_sm).fit(cov_type='hc2')
        
        # Validate against statsmodels
        comparison = validate_against_statsmodels(X, y, cov_type='HC2')
        
        # Check that the differences are small
        coef_diff = np.max(np.abs(comparison['rel_coef_diff']))
        se_diff = np.max(np.abs(comparison['rel_se_diff']))
        
        # Print the summary
        print("\nOur implementation summary:")
        print(rse.summary())
        
        # Print statsmodels summary for comparison
        print("\nStatsmodels summary:")
        coef_names = ['intercept'] + [f'x{i}' for i in range(k)]
        sm_summary = pd.DataFrame({
            'Coefficient': sm_model.params,
            'Std Error (HC2)': sm_model.bse,
            't value': sm_model.tvalues,
            'P>|t|': sm_model.pvalues
        }, index=coef_names)
        print(sm_summary)
        
        # Print detailed comparison
        print("\nDetailed comparison:")
        comparison_df = pd.DataFrame({
            'Our Coef': rse.coef_,
            'SM Coef': sm_model.params,
            'Coef Diff': comparison['coef_difference'],
            'Our SE': rse.std_errors_,
            'SM SE': sm_model.bse,
            'SE Diff': comparison['se_difference'],
            'Rel SE Diff %': comparison['rel_se_diff'] * 100
        }, index=coef_names)
        print(comparison_df)
        
        # Print validation results summary
        print("\nValidation summary:")
        print(f"Maximum relative coefficient difference: {coef_diff:.10f}")
        print(f"Maximum relative standard error difference: {se_diff:.10f}")
        
        # Set stricter tolerance for test pass
        tolerance = 1e-6
        if coef_diff < tolerance and se_diff < tolerance:
            print(f"\n✓ Test passed! Differences are within tolerance ({tolerance}).")
        else:
            print(f"\n✗ Test failed: Differences exceed tolerance ({tolerance}).")
            
    except ImportError:
        print("\nstatsmodels not installed. Skipping validation.")
        print("\nImplementation summary:")
        print(rse.summary())

#%% Usage Examples

def example_basic_usage():
    """Basic usage example with heteroskedastic data."""
    print("\n=== Basic Usage Example ===")
    
    # Generate heteroskedastic data
    np.random.seed(42)
    n = 200
    X = np.random.normal(0, 1, (n, 3))
    X_df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
    
    # True coefficients
    beta = np.array([1.5, -0.8, 0.3])
    
    # Generate heteroskedastic errors (variance increases with feature1)
    heteroskedasticity = np.exp(X[:, 0])
    errors = np.random.normal(0, 1, n) * heteroskedasticity
    y = X @ beta + errors
    
    # Fit ordinary least squares model
    model = LinearRegression(fit_intercept=True).fit(X_df, y)
    
    # Get standard OLS summary (which does not account for heteroskedasticity)
    y_pred = model.predict(X_df)
    residuals = y - y_pred
    rss = (residuals**2).sum()
    mse = rss / (n - 4)  # 4 = 3 features + intercept
    
    # Get OLS standard errors (naive approach)
    X_with_intercept = np.column_stack([np.ones(n), X])
    cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    ols_se = np.sqrt(np.diag(cov_matrix))
    
    # Calculate robust standard errors (HC0, HC1, HC2, HC3)
    rse_results = {}
    for cov_type in ['HC0', 'HC1', 'HC2', 'HC3']:
        rse = RobustStandardErrors(model, X_df, y, cov_type=cov_type).fit()
        rse_results[cov_type] = rse
    
    # Print results comparison
    coef_with_intercept = np.append(model.intercept_, model.coef_)
    
    print("\nCoefficient comparison:")
    comparison = pd.DataFrame({
        'Coefficient': coef_with_intercept,
        'OLS SE': ols_se,
        'HC0 SE': rse_results['HC0'].std_errors_,
        'HC1 SE': rse_results['HC1'].std_errors_,
        'HC2 SE': rse_results['HC2'].std_errors_,
        'HC3 SE': rse_results['HC3'].std_errors_
    }, index=['intercept', 'feature1', 'feature2', 'feature3'])
    
    print(comparison)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    # Plot residuals vs feature1 to show heteroskedasticity
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], residuals, alpha=0.5)
    plt.title('Residuals vs feature1')
    plt.xlabel('feature1')
    plt.ylabel('Residuals')
    
    # Plot standard error comparison
    plt.subplot(1, 2, 2)
    
    # Prepare data for plotting
    coef_names = ['intercept', 'feature1', 'feature2', 'feature3']
    se_data = pd.DataFrame({
        'OLS': ols_se,
        'HC0': rse_results['HC0'].std_errors_,
        'HC1': rse_results['HC1'].std_errors_,
        'HC2': rse_results['HC2'].std_errors_,
        'HC3': rse_results['HC3'].std_errors_
    }, index=coef_names)
    
    se_data.plot(kind='bar', ax=plt.gca())
    plt.title('Standard Error Comparison')
    plt.ylabel('Standard Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    print("\nPlot shows the difference between ordinary SEs and robust SEs.")
    print("Note how robust standard errors are larger, accounting for heteroskedasticity.")
    plt.show()
    
def example_with_regularization():
    """Example with regularized models (Ridge and Lasso)."""
    print("\n=== Regularized Models Example ===")
    
    # Generate data with some irrelevant features
    X, y = make_regression(n_samples=200, n_features=20, n_informative=5, 
                         noise=20, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train different models
    models = {
        'OLS': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1)
    }
    
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        
        # Calculate test score
        test_score = model.score(X_test, y_test)
        
        # Calculate robust standard errors
        rse = RobustStandardErrors(model, X_train, y_train, cov_type='HC2').fit()
        
        # Print summary
        print(f"\n{name} Model (R² on test: {test_score:.3f}):")
        
        # Print coefficients and SEs for non-zero coefficients
        summary = rse.summary()
        if name == 'Lasso':
            print("Showing only non-zero coefficients:")
            non_zero_mask = np.abs(rse.coef_) > 1e-10
            print(summary[non_zero_mask])
        else:
            # For OLS and Ridge, show top 5 most significant coefficients
            top_indices = np.argsort(np.abs(rse.t_stats_))[-5:]
            print("Showing top 5 most significant coefficients:")
            print(summary.iloc[top_indices])


def example_with_dask():
    """Example using dask arrays for large-scale regression analysis."""
    if not HAS_DASK:
        print("Dask is not installed. Skipping example.")
        return
    
    try:
        import dask.array as da
        from dask_ml.linear_model import LinearRegression as DaskLinearRegression
    except ImportError:
        print("dask-ml is not installed. Skipping example.")
        return
    
    print("Running example with dask arrays...")
    
    # Generate a larger dataset
    np.random.seed(42)
    n = 10000  # 10,000 observations
    k = 10     # 10 features
    
    # Generate data in chunks to simulate large dataset
    chunk_size = 1000
    X_np = np.random.normal(0, 1, (n, k))
    beta = np.random.uniform(-1, 1, k)
    
    # Create heteroskedastic errors
    e = np.random.normal(0, 1, n) * np.abs(X_np[:, 0])
    y_np = X_np @ beta + e
    
    # Convert to dask arrays with chunks
    X = da.from_array(X_np, chunks=(chunk_size, k))
    y = da.from_array(y_np, chunks=(chunk_size,))
    
    print(f"Created dask arrays with shape={X.shape}, chunks={X.chunks}")
    
    # Fit a model using dask-ml
    print("Fitting model with dask-ml...")
    model = DaskLinearRegression(fit_intercept=True).fit(X, y)
    
    # Calculate robust standard errors
    print("Calculating HC2 robust standard errors...")
    rse = RobustStandardErrors(model, X, y, cov_type='HC2').fit()
    
    # Print the summary
    print("\nRobust standard errors summary (with dask):")
    print(rse.summary())
    
    # Check model coefficients vs true coefficients
    print("\nTrue vs estimated coefficients:")
    comparison_df = pd.DataFrame({
        'True': np.append(0, beta),  # Add 0 for intercept
        'Estimated': rse.coef_,
        'Std Error (HC2)': rse.std_errors_,
        'p-value': rse.p_values_
    })
    comparison_df.index = rse.feature_names
    print(comparison_df)

#%% Main Guard

if __name__ == "__main__":
    test_robust_standard_errors()
    example_basic_usage()
    example_with_regularization()
    example_with_dask()