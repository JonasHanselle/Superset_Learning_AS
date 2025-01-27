import numpy as np
import copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from .neural_net_superset import NeuralNetworkGLM
from scipy.stats import truncnorm
import scipy


def impute_censored(X, y, censored, base_model, distr_func, upper_threshold, max_runs=30, max_abs=1e+00):
    """Iterative imputation of censored values after Schmee & Hahn (1979)

    Iteratively fits the base model while imputing the censored values through mean estimates of a truncated normal distribution.
    """
    # numerical issues arise for normal distribution if a and b are too far apart
    # these are resolved manually later (the conditional mean is set to the timeout)
    #np.seterr(invalid='ignore')

    model = copy.deepcopy(base_model)
    model.fit(X[~censored], y[~censored])

    for i in range(max_runs):
        mean, std = distr_func(model, X[censored], y[censored])
        rv = truncnorm(a=(upper_threshold - mean) / std,
                       b=(upper_threshold - mean) / std + 1*std, loc=mean, scale=std)
        cmean = rv.mean()
        print("cmean", cmean)
        mask = np.isnan(cmean) | np.isposinf(cmean)

        if mask.any():
            cmean[mask] = np.squeeze(y[censored][mask,:]) 

        # cmean

        max_diff = np.max(np.abs(y[censored] - cmean))
        print("y shape", y.shape)
        print("censored shape", censored.shape)
        print("y censored shape", y[censored].shape)
        print("cmean shape", cmean.shape)
        y[censored] = cmean


        if max_diff < max_abs:
            break

        model = copy.deepcopy(base_model)
        print(f"imputation iteration: {i}")
        model.fit(X, y)

    return model

def distr_func(model, X, y=None):
    if isinstance(model, RandomForestRegressor):
        predictions = np.empty(shape=(X.shape[0], model.n_estimators))

        for num, tree in enumerate(range(model.n_estimators)):
            predictions[:, num] = model.estimators_[tree].predict(X)

        return np.mean(predictions, axis=1), np.std(predictions, axis=1)

    elif isinstance(model, Ridge):
        mean = model.predict(X)
        std = np.sum((y - mean)**2)
        std = np.sqrt((1 / (y.size - 1)) * std)
        return mean, std

    elif isinstance(model, NeuralNetworkGLM):
        print("instance correct")
        mean = np.squeeze(model.predict(X)) 
        std = np.sum((y - mean)**2)
        std = np.sqrt((1 / (y.size - 1)) * std)
        return mean, std

    raise NotImplementedError()