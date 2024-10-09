import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from age_matcher.age_matcher import AgeMatcher


def test_metrics():
    matcher = AgeMatcher()
    x = np.random.rand(100)
    matcher.matches['age_diff'] = x
    metrics = matcher._calc_metrics()
    assert metrics['mse'] == mean_squared_error(x, np.zeros_like(x))
    assert metrics['mae'] == mean_absolute_error(x, np.zeros_like(x))
