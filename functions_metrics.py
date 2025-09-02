import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_forecasts(y_true, y_pred, q=0.9):
    """
    Evaluate forecasts at multiple levels using R2, MASE, and peak-weighted RMSE.
    
    Args:
        y_true : np.ndarray of shape (n_samples, n_days, n_features)
        y_pred : np.ndarray of shape (n_samples, n_days, n_features)
        q      : float, quantile for peak-weighted RMSE (default=0.9 for top-10%)
    
    Returns:
        results : dict
            {
            "global": { "R2": float, "MASE": float, "PeakRMSE": float },
            "per_feature": { "R2": np.array[n_features], "MASE": ..., "PeakRMSE": ... },
            "per_day": { "R2": np.array[n_days], "MASE": ..., "PeakRMSE": ... }
            }
    """
    assert y_true.shape == y_pred.shape, "Shapes must match!"
    n_samples, n_days, n_features = y_true.shape
    
    # flatten everything for global metrics
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    global_r2   = r2_score(y_true_flat, y_pred_flat)
    global_mase = mase(y_true, y_pred)  # uses full 3D arrays
    global_peak = peak_weighted_rmse(y_true_flat, y_pred_flat, q=q)
    
    # per feature (aggregate across samples and days)
    per_feature_r2   = []
    per_feature_mase = []
    per_feature_peak = []
    for f in range(n_features):
        yt = y_true[:,:,f].reshape(-1)
        yp = y_pred[:,:,f].reshape(-1)
        per_feature_r2.append(r2_score(yt, yp))
        per_feature_mase.append(mase(y_true[:,:,f][:,:,None], 
                                     y_pred[:,:,f][:,:,None]))
        per_feature_peak.append(peak_weighted_rmse(yt, yp, q=q))
    per_feature = {
        "R2": np.array(per_feature_r2),
        "MASE": np.array(per_feature_mase),
        "PeakRMSE": np.array(per_feature_peak)
    }
    
    # per day (aggregate across samples and features)
    per_day_r2   = []
    per_day_mase = []
    per_day_peak = []
    for d in range(n_days):
        yt = y_true[:,d,:].reshape(-1)
        yp = y_pred[:,d,:].reshape(-1)
        per_day_r2.append(r2_score(yt, yp))
        per_day_mase.append(mase(y_true[:,d,:][:,None,:], 
                                 y_pred[:,d,:][:,None,:]))
        per_day_peak.append(peak_weighted_rmse(yt, yp, q=q))
    per_day = {
        "R2": np.array(per_day_r2),
        "MASE": np.array(per_day_mase),
        "PeakRMSE": np.array(per_day_peak)
    }
    
    return {
        "global": {"R2": global_r2, "MASE": global_mase, "PeakRMSE": global_peak},
        "per_feature": per_feature,
        "per_day": per_day
    }

def mase(y, yhat):
    """Calculates the Mean Absolute Scaled Error (MASE) between actual and predicted values.

    MASE is a scale-independent metric for evaluating forecast accuracy, allowing comparison across different datasets.

    Args:
        y (np.ndarray): Actual values of shape (n_samples, n_timesteps, n_features).
        yhat (np.ndarray): Predicted values of shape (n_samples, n_timesteps, n_features).

    Returns:
        float: The Mean Absolute Scaled Error (MASE) value.

    Notes:
        - Checks that y and yhat have the same shape and are 3D arrays.
        - If the mean absolute difference of consecutive actual values is zero, the denominator is set to 1.0 to avoid division by zero.
        - Returns np.nan if input shapes are invalid.
    """
    # Check input shapes
    if not (isinstance(y, np.ndarray) and isinstance(yhat, np.ndarray)):
        raise TypeError("Inputs y and yhat must be numpy arrays.")
    if y.shape != yhat.shape:
        raise ValueError("Shapes of y and yhat must match.")
    if y.ndim != 3:
        raise ValueError("Inputs y and yhat must be 3D arrays (n_samples, n_timesteps, n_features).")

    # we use a mask for any nan values
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    errs = np.abs(y - yhat)
    errs = np.where(mask, errs, np.nan)

    # per-series denominator: naive forecast error along time
    diffs = np.abs(np.diff(y, axis=1))
    denom = np.nanmean(diffs, axis=1)  # mean over time
    denom = np.where(denom == 0, 1.0, denom)  # avoid /0

    # per-series numerator: MAE over valid time<
    num = np.nanmean(errs, axis=1)

    # average across (sample, feature)
    return np.nanmean(num / denom[:, None])

def peak_weighted_rmse(y, yhat, q=0.9, alpha=4.0):
    """peak-weighted RMSE over batched multivariate series.

    This metric gives higher weight to errors occurring at the peaks of the true values, as determined by a specified quantile.

    Args:
        y (array-like): True values.
        yhat (array-like): Predicted values.
        q (float, optional (default=0.9)): Quantile threshold to define peaks in `y`.
        alpha (float, optional (default=4.0)): Weighting factor applied to peak errors.

    Returns:
        float: Peak-weighted RMSE value.
    """
    # we use a mask for any nan values
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y, yhat = y[mask], yhat[mask]
    
    if len(y) == 0:   # all NaN case
        return np.nan
    
    # calculation happens here
    qv = np.quantile(y, q)
    w = 1.0 + alpha * (y >= qv).astype(float)
    return np.sqrt(((w * (y - yhat)**2).sum()) / w.sum())

def delta_mae(y, yhat):
    """
    Calculates the mean absolute error (MAE) between the first differences of two sequences.

    Args:
        y (array-like): The true values sequence.
        yhat (array-like): The predicted values sequence.

    Returns:
        float: The mean absolute error between the first differences of `y` and `yhat`.
    """
    dy, dyhat = np.diff(y), np.diff(yhat)
    return np.abs(dy - dyhat).mean()