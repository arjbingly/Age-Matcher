import logging
import math
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind


class AgeMatcher:
    """Class to match cases to controls based on age and optionally sex.

    Attributes:
       cases_df (pd.DataFrame): DataFrame containing the cases.
       controls_df (pd.DataFrame): DataFrame containing the controls.
       age_tol (float): Age tolerance for matching.
       age_col (str): Column name for age.
       sex_col (Optional[str]): Column name for sex.
       strategy (Literal['greedy', 'stricter']): Matching strategy to use.
       shuffle_df (bool): Whether to shuffle the DataFrames before matching.
       random_state (Optional[int]): Random state for shuffling.
       convert_age_to_int (bool): Whether to convert age to integer.
       verbose (Union[bool, int]): Verbosity level for logging.
       matches (Dict[str, List[Union[int, float]]]): Dictionary to store matches.
       unmatched_cases (List[int]): List of unmatched case IDs.
       _info (dict): Dictionary to store information about the matching process.

    Notes:
        The class assumes that the cases and controls dataframes have the same column names for age and optionally sex.
        The class assumes that in the worst case, the number of cases is lesser than the number of controls.
        The age_tol is converted to upper ceiling integer for the stricter strategy.

    Example:
        ```python
        import pandas as pd
        from age_matcher.age_matcher import AgeMatcher

        # Create example data
        cases_data = {
            'id': [1, 2, 3, 4],
            'age': [25, 35, 45, 55],
            'sex': ['M', 'F', 'M', 'F']
        }
        controls_data = {
            'id': [5, 6, 7, 8, 9, 10],
            'age': [26, 36, 46, 56, 30, 40],
            'sex': ['M', 'F', 'M', 'F', 'M', 'F']
        }

        cases_df = pd.DataFrame(cases_data).setIndex('id')
        controls_df = pd.DataFrame(controls_data).setIndex('id')

        # Initialize AgeMatcher
        matcher = AgeMatcher(age_tol=5, age_col='age', sex_col='sex', strategy='greedy', shuffle_df=True, random_state=42)

        # Perform matching
        matched_cases, matched_controls = matcher(cases_df, controls_df)

        # Display results
        print("Matched Cases:")
        print(matched_cases)
        print("Matched Controls:")
        print(matched_controls)
        ```
    """

    def __init__(self, age_tol: float = 3, age_col: str = 'age', sex_col: Optional[str] = 'sex',
                 strategy: Literal['greedy', 'stricter'] = 'stricter', shuffle_df: bool = True,
                 random_state: Optional[int] = None, convert_age_to_int: bool = True, verbose: Union[bool, int] = True):
        """Initializes the AgeMatcher with the given parameters.

        Args:
            age_tol (float): Age tolerance for matching. Defaults to 3.
            age_col (str): Column name for age. Defaults to 'age'.
            sex_col (Optional[str]): Column name for sex. Defaults to 'sex'. If None does not match based on sex.
            strategy (Literal['greedy', 'stricter']): Matching strategy to use. Defaults to 'stricter'.
            shuffle_df (bool): Whether to shuffle the DataFrames before matching. Defaults to True.
            random_state (Optional[int]): Random state for shuffling. Defaults to None.
            convert_age_to_int (bool): Whether to convert age to integer. Defaults to True.
            verbose (Union[bool, int]): Verbosity level for logging. Defaults to True.
        """
        self.setup_logger(verbose)

        self.age_tol = age_tol

        self.age_col = age_col
        self.sex_col = sex_col

        self.strategy = strategy
        self.shuffle_df = shuffle_df
        self.random_state = random_state

        self.convert_age_to_int = convert_age_to_int

        self.cases_df = pd.DataFrame()
        self.controls_df = pd.DataFrame()
        self.matches: Dict[str, List[Union[int, float]]] = {'case_id': [], 'control_id': [], 'age_diff': []}
        self.unmatched_cases: List[int] = []
        self._info: dict = {}

    def __call__(self, *args, **kwargs):
        """Calls the match method to perform the matching."""
        return self.match(*args, **kwargs)

    def _verify_col(self, col):
        """Verifies that the given column exists in both cases and controls DataFrames.

        Args:
            col (str): Column name to verify.

        Raises:
            ValueError: If the column is not found in either DataFrame.
        """
        if col not in self.cases_df.columns:
            self.logger.error(f"Provided '{col}' not found in cases dataframe")
            raise ValueError(f"Provided '{col}' not found in cases dataframe")
        if col not in self.controls_df.columns:
            self.logger.error(f"Provided '{col}' not found in controls dataframe")
            raise ValueError(f"Column '{col}' not found in controls dataframe")

    def setup_logger(self, verbose):
        """Sets up the logger with the given verbosity level.

        Args:
            verbose (Union[bool, int]): Verbosity level for logging.

        Raises:
            ValueError: If verbose is not a boolean or an integer.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        if isinstance(verbose, bool):
            self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
        elif isinstance(verbose, int):
            self.logger.setLevel(verbose)
        else:
            raise ValueError("verbose must be a boolean or an integer")

    def _convert_age_to_int(self):
        """Converts the age column to integer type in both cases and controls DataFrames."""
        self.cases_df[self.age_col] = self.cases_df[self.age_col].astype(int)
        self.controls_df[self.age_col] = self.controls_df[self.age_col].astype(int)

    def _shuffle_df(self):
        """Shuffles the cases and controls DataFrames."""
        self.cases_df = self.cases_df.sample(frac=1, random_state=self.random_state)
        self.controls_df = self.controls_df.sample(frac=1, random_state=self.random_state)

    def _add_match(self, case_id, control_id, age_diff):
        """Adds a match to the matches dictionary.

        Args:
            case_id (int): ID of the case.
            control_id (int): ID of the control.
            age_diff (float): Age difference between the case and control.
        """
        self.matches['case_id'].append(int(case_id))
        self.matches['control_id'].append(int(control_id))
        self.matches['age_diff'].append(float(age_diff))

    def _greedy_match(self, cases: pd.DataFrame, controls: pd.DataFrame):
        """Performs greedy matching of cases to controls based on age tolerance.

        Args:
            cases (pd.DataFrame): DataFrame containing the cases.
            controls (pd.DataFrame): DataFrame containing the controls.
        """
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
                self.unmatched_cases.append(case_idx)  # TODO: Log warning

    def _stricter_match(self, cases: pd.DataFrame, controls: pd.DataFrame):
        """Performs matching of cases to controls, more stringent than greedy matching.

        Args:
            cases (pd.DataFrame): DataFrame containing the cases.
            controls (pd.DataFrame): DataFrame containing the controls.

        Notes:
            Uses the upper cieling of the age tolerance to match cases to controls.
        """
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

    def match(self, cases_df: pd.DataFrame, controls_df: pd.DataFrame):
        """Matches cases to controls based on the specified strategy.

        Args:
            cases_df (pd.DataFrame): DataFrame containing the cases.
            controls_df (pd.DataFrame): DataFrame containing the controls.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames of matched cases and controls.
        """
        self.cases_df = cases_df
        self.controls_df = controls_df

        self._verify_col(self.age_col)
        if self.sex_col is not None:
            self._verify_col(self.sex_col)

        if self.convert_age_to_int:
            self._convert_age_to_int()

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
        """Performs the matching of cases to controls based on the specified strategy.

        Args:
            cases (pd.DataFrame): DataFrame containing the cases.
            controls (pd.DataFrame): DataFrame containing the controls.
            strategy (str): Matching strategy to use ('greedy' or 'stricter').

        Raises:
            ValueError: If an invalid matching strategy is provided.
        """
        match strategy:
            case 'greedy':
                self._greedy_match(cases, controls)
            case 'stricter':
                self._stricter_match(cases, controls)
            case _:
                raise ValueError(f"Invalid matching strategy: {strategy}, must be 'greedy' or 'stricter'")

    def get_matched_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Retrieves the matched cases and controls DataFrames.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames of matched cases and controls.
        """
        matched_cases = self.cases_df.loc[self.matches['case_id'], :]
        matched_controls = self.controls_df.loc[self.matches['control_id'], :]
        return matched_cases, matched_controls

    def get_info(self):
        """Gathers information about the matching process and updates the _info attribute."""
        self._info.update({'num_cases': len(self.cases_df), 'num_controls': len(self.controls_df),
                           'num_matched': len(self.matches['case_id']), })
        self._info.update(self._calc_metrics())
        self._info.update(self._calc_stats())

    def log_info(self):
        """Logs information about the matching process."""
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
        """Calculates the mean absolute error and mean squared error of the matches.

        Returns:
            dict: Dictionary containing the MAE and MSE.
        """
        mae = np.mean(np.abs(self.matches['age_diff']))
        mse = np.mean(np.square(self.matches['age_diff']))
        return {'mae': mae, 'mse': mse}

    def _calc_stats(self):
        """Calculates statistical tests (T-test and Kolmogorov–Smirnov test) on the matched data.

        Returns:
            dict: Dictionary containing the T-test and Kolmogorov–Smirnov test statistics and p-values.
        """
        matched_cases, matched_controls = self.get_matched_data()
        ttest_stat, ttest_pval = ttest_ind(matched_cases[self.age_col], matched_controls[self.age_col],
                                           alternative='two-sided')
        ks_stat, ks_pval = ks_2samp(matched_cases[self.age_col], matched_controls[self.age_col],
                                    alternative='two-sided')
        return {'ttest_stat': ttest_stat, 'ttest_pval': ttest_pval, 'ks_stat': ks_stat, 'ks_pval': ks_pval, }
