import logging
import math
from typing import List, Optional, Tuple, Union, Literal, Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind


class AgeMatcher:
    def __init__(self,
                 cases_df: pd.DataFrame,
                 controls_df: pd.DataFrame,
                 age_tol: float = 3, # Note that the tol is converted to int for the sivan strategy
                 age_col: str = 'age',
                 sex_col: Optional[str] = 'sex',
                 strategy: Literal['greedy', 'sivan'] = 'greedy',
                 shuffle_df: bool = True,
                 random_state: Optional[int] = None,
                 convert_age_to_int: bool = True,
                 verbose: Union[bool, int] = True):

        self.setup_logger(verbose)
        self.cases_df = cases_df
        self.controls_df = controls_df
        self.age_tol = age_tol

        self.age_col = age_col
        self._verify_col(self.age_col)
        self.sex_col = sex_col
        if self.sex_col is not None:
            self._verify_col(self.sex_col)

        self.strategy = strategy
        self.shuffle_df = shuffle_df
        self.random_state = random_state

        if convert_age_to_int:
            self._convert_age_to_int()

        self.matches: Dict[str, List[Union[int,float]]] = {'case_id': [], 'control_id': [], 'age_diff': []}
        self.unmatched_cases: List[int] = []
        self._info: dict = {}

    def __call__(self):
        return self.match()

    def _verify_col(self, col):
        if col not in self.cases_df.columns:
            self.logger.error(f"Provided '{col}' not found in cases dataframe")
            raise ValueError(f"Provided '{col}' not found in cases dataframe")
        if col not in self.controls_df.columns:
            self.logger.error(f"Provided '{col}' not found in controls dataframe")
            raise ValueError(f"Column '{col}' not found in controls dataframe")

    def setup_logger(self, verbose):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        if isinstance(verbose, bool):
            self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
        elif isinstance(verbose, int):
            self.logger.setLevel(verbose)
        else:
            raise ValueError("verbose must be a boolean or an integer")
    def _convert_age_to_int(self):
        self.cases_df[self.age_col] = self.cases_df[self.age_col].astype(int)
        self.controls_df[self.age_col] = self.controls_df[self.age_col].astype(int)

    def _shuffle_df(self):
        self.cases_df = self.cases_df.sample(frac=1, random_state=self.random_state)
        self.controls_df = self.controls_df.sample(frac=1, random_state=self.random_state)

    def _add_match(self, case_id, control_id, age_diff):
        self.matches['case_id'].append(int(case_id))
        self.matches['control_id'].append(int(control_id))
        self.matches['age_diff'].append(float(age_diff))

    def _greedy_match(self, cases: pd.DataFrame, controls: pd.DataFrame):
        used_controls = set()
        for case_idx, case in cases.iterrows():
            available_controls = controls[~controls.index.isin(used_controls)].copy()
            available_controls['_age_diff'] = abs(available_controls[self.age_col] - case[self.age_col])
            if len(available_controls) == 0:
                return  # No more controls to match
            matched_control = available_controls.nsmallest(1, '_age_diff', keep='first')
            if matched_control['_age_diff'].values[0] <= self.age_tol:
                used_controls.add(int(matched_control.index[0]))
                self._add_match(case_idx, matched_control.index[0], matched_control['_age_diff'].values[0])
            else:
                self.unmatched_cases.append(case_idx)
                # TODO: Log warning

    def _sivan_match(self, cases: pd.DataFrame, controls: pd.DataFrame):
        """Uses upper ciel of age tolerance to match cases to controls."""
        used_controls = set()
        used_cases = set()

        def _find_unmatched_cases():
            self.unmatched_cases.extend(list(cases[~cases.index.isin(used_cases)].index))

        for age_diff in range(1 + int(math.ceil(self.age_tol))):
            available_cases = cases[~cases.index.isin(used_cases)].copy()
            if len(available_cases) == 0:
                _find_unmatched_cases()
                return  # No more cases to match
            for case_idx, case in available_cases.iterrows():
                available_controls = controls[~controls.index.isin(used_controls)].copy()
                if len(available_controls) == 0:
                    _find_unmatched_cases()
                    return  # No more controls to match
                available_controls['_age_diff'] = abs(available_controls[self.age_col] - case[self.age_col])
                matched_control = available_controls.nsmallest(1, '_age_diff', keep='first')
                if matched_control['_age_diff'].values[0] <= age_diff:
                    used_controls.add(int(matched_control.index[0]))
                    used_cases.add(case_idx)
                    self._add_match(case_idx, matched_control.index[0], matched_control['_age_diff'].values[0])

    def match(self):
        if self.shuffle_df:
            self._shuffle_df()
        if self.sex_col is not None:
            for sex in self.cases_df[self.sex_col].unique():
                cases = self.cases_df[self.cases_df[self.sex_col] == sex].copy()
                controls = self.controls_df[self.controls_df[self.sex_col] == sex].copy()
                self._match(cases, controls, self.strategy)
        else:
            self._match(self.cases_df, self.controls_df, self.strategy)
        self.get_info()
        self.log_info()
        return self.get_matched_data()

    def _match(self, cases, controls, strategy: str = 'greedy'):
        """Performs the matching of cases to controls based on age tolerance."""
        match strategy:
            case 'greedy':
                self._greedy_match(cases, controls)
            case 'sivan':
                self._sivan_match(cases, controls)
            case _:
                raise ValueError(f"Invalid matching strategy: {strategy}, must be 'greedy' or 'sivan'")

    def get_matched_data(self) -> pd.DataFrame:
        matched_cases = self.cases_df.loc[self.matches['case_id'], :]
        matched_controls = self.controls_df.loc[self.matches['control_id'], :]
        return matched_cases, matched_controls

    def get_info(self):
        self._info.update({'num_cases': len(self.cases_df),
                           'num_controls': len(self.controls_df),
                           'num_matched': len(self.matches['case_id']), })
        self._info.update(self._calc_metrics())
        self._info.update(self._calc_stats())

    def log_info(self):
        self.logger.info(f"Number of cases: {self._info['num_cases']}")
        self.logger.info(f"Number of controls: {self._info['num_controls']}")
        self.logger.info(f"Number of matched: {self._info['num_matched']}")
        self.logger.info(f"Mean Absolute Error: {self._info['mae']:.4f}")
        self.logger.info(f"Mean Squared Error: {self._info['mse']:.4f}")
        self.logger.info(f"T-test statistic: {self._info['ttest_stat']:.4f}")
        self.logger.info(f"T-test p-value: {self._info['ttest_pval']:.4f}")
        self.logger.info(f"KS statistic: {self._info['ks_stat']:.4f}")
        self.logger.info(f"KS p-value: {self._info['ks_pval']:.4f}")

    def _calc_metrics(self):
        mae = np.mean(np.abs(self.matches['age_diff']))
        mse = np.mean(np.square(self.matches['age_diff']))
        return {'mae': mae, 'mse': mse}

    def _calc_stats(self):
        matched_cases, matched_controls = self.get_matched_data()
        ttest_stat, ttest_pval = ttest_ind(matched_cases[self.age_col], matched_controls[self.age_col],
                                           alternative='two-sided')
        ks_stat, ks_pval = ks_2samp(matched_cases[self.age_col], matched_controls[self.age_col],
                                    alternative='two-sided')
        return {
            'ttest_stat': ttest_stat,
            'ttest_pval': ttest_pval,
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
        }
