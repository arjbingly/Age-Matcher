import pytest
import pandas as pd
import numpy as np

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

@pytest.mark.parametrize("col_name", ['age', 'sex'])
def test_verify_col(col_name):
    # Good Case
    cases, controls = get_dummy_data()
    assert AgeMatcher(cases, controls), "Should not raise an error"
    cases_wrong = cases.rename(columns={col_name: 'wrong_col'})
    controls_wrong = controls.rename(columns={col_name: 'wrong_col'})
    # wrong age_col
    with pytest.raises(ValueError):
        AgeMatcher(cases_wrong, controls)
    # wrong sex_col
    with pytest.raises(ValueError):
        AgeMatcher(cases, controls_wrong, age_col='wrong_col')

@pytest.mark.parametrize(
    "dtype", [str, float])
def test_convert_age_to_int(dtype):
    cases, controls = get_dummy_data()
    cases = cases.astype({'age': dtype})
    matcher = AgeMatcher(cases, controls, convert_age_to_int=True)
    assert matcher.cases_df['age'].dtype == 'int64', "Should convert age to int"
    controls = controls.astype({'age': dtype})
    matcher = AgeMatcher(cases, controls, convert_age_to_int=True)
    assert matcher.controls_df['age'].dtype == 'int64', "Should convert age to int"

def test_shuffle_df():
    cases, controls = get_dummy_data()
    matcher = AgeMatcher(cases.copy(), controls.copy(), shuffle_df=True, random_state=42)
    cases = cases.sample(frac=1, random_state=42)
    controls = controls.sample(frac=1, random_state=42)
    assert not matcher.cases_df.equals(cases), "Should shuffle cases"
    assert not matcher.controls_df.equals(controls), "Should shuffle controls"

def test_add_match():
    cases, controls = get_dummy_data()
    matcher = AgeMatcher(cases, controls)
    matcher._add_match(1, 1, 0)
    assert matcher.matches == {'case_id': [1], 'control_id': [1], 'age_diff': [0]}, "Should add match"

