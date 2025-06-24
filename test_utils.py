import pytest
import pandas as pd
import numpy as np
import utils

@pytest.fixture
def df_sample():
    return pd.DataFrame({
        'num1': [1, 2, 3, 4, 100],
        'num2': [10, 20, np.nan, 40, 50],
        'cat': ['A', 'B', 'A', 'C', 'B'],
        'actual_productivity': [1.1, 2.2, 3.3, 4.4, 5.5]
    })

def test_handle_missing(df_sample):
    df = utils.handle_missing_values(df_sample.copy())
    assert df['num2'].isnull().sum() == 0

def test_outlier_treatment(df_sample):
    df = utils.treat_outliers_iqr(df_sample.copy(), ['num1'])
    assert df['num1'].max() <= df_sample['num1'].quantile(0.75) + 1.5 * (df_sample['num1'].quantile(0.75) - df_sample['num1'].quantile(0.25))

def test_dummies(df_sample):
    df = utils.convert_to_dummies(df_sample.copy(), ['cat'])
    assert 'cat_B' in df.columns
    assert 'cat_C' in df.columns
    assert 'cat' not in df.columns
