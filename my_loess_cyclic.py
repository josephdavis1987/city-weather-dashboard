import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

def loess_cyclic(x, y, frac=0.1):
    """
    LOESS smoothing on a cyclic domain.
    - x: 1D array of day-of-year (1…n)
    - y: measurements (same length)
    - frac: LOESS span (fraction of points)
    Returns y_smoothed of length n.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    # number of neighbors ≈ frac * n
    k = max(int(frac * n), 1)

    # build extended arrays: last k days mapped to -(n-k…n), then original, then first k days mapped to (n+1…n+k)
    x_ext = np.concatenate((x[-k:] - n, x, x[:k] + n))
    y_ext = np.concatenate((y[-k:], y, y[:k]))

    # run LOESS on the extended series
    sm_ext = lowess(y_ext, x_ext, frac=frac)
    # sm_ext is sorted by x_ext, so sm_ext[:,1] aligns with x_ext

    # extract just the middle block back
    y_sm = sm_ext[k:k+n, 1]
    return y_sm
