import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from age_matcher.age_matcher import AgeMatcher
def get_dummy_data():
    age_cases = np.arange(20, 41, 4)
    age_controls = np.arange(20, 41, 2)


    cases = pd.DataFrame(
        {'id': np.arange(1, len(age_cases)+1),
         'age': age_cases,
         'sex': ['M' for _ in range(len(age_cases))]}
    ).set_index('id')

    controls = pd.DataFrame(
        {'id': np.arange(1, len(age_controls)+1),
         'age': age_controls,
         'sex': ['M' for _ in range(len(age_controls))]}
    ).set_index('id')
    return cases, controls

def test_metrics():
    cases, controls = get_dummy_data()
    matcher = AgeMatcher(cases, controls)
    x  = np.random.rand(100)
    matcher.matches['age_diff'] = x
    metrics = matcher._calc_metrics()
    assert metrics['mse'] == mean_squared_error(x, np.zeros_like(x))
    assert metrics['mae'] == mean_absolute_error(x, np.zeros_like(x))