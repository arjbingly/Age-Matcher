# Age Matcher

## Overview

Age Matcher is a Python library designed to match cases to controls based on age and optionally sex. It provides flexible matching strategies and ensures that the matching process is efficient and accurate.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- Match cases to controls based on age tolerance.
- Optionally match based on sex.
- Supports different matching strategies: `greedy` and `sricter`.
- Shuffle DataFrames before matching for randomness.
- Metrics and statistics about the matching process.
- Detailed logging and information about the matching process.

## Requirements

- Python 3.6+
- Pandas
- NumPy
- SciPy

## Installation

The package is still in development and not yet available on PyPI. 
You can install it directly from the GitHub repository using `pip`:

```bash
pip install git+https://github.com/arjbingly/Age-Matcher.git
```

## Usage

Here's a basic example of how to use the Age Matcher library:

```python
import pandas as pd
from age_matcher import AgeMatcher

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

cases_df = pd.DataFrame(cases_data).set_index('id')
controls_df = pd.DataFrame(controls_data).set_index('id')

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
Expected output:
```console
Number of cases: 4
Number of controls: 6
Number of matched: 4
Mean Absolute Error: 1.0000
Mean Squared Error: 1.0000
T-test statistic: -0.1095
T-test p-value: 0.9163
KS statistic: 0.2500
KS p-value: 1.0000
Matched Cases:
    age sex
id         
2    35   F
4    55   F
1    25   M
3    45   M
Matched Controls:
    age sex
id         
6    36   F
8    56   F
5    26   M
7    46   M
```

## Testing

To run the tests, you can use `pytest`:

```bash
pytest src/tests
```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

Please ensure your code follows the project's coding standards and includes tests for any new features or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or suggestions, please open an issue or contact the repository owner.
