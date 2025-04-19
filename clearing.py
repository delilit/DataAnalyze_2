import numpy as np
import pandas as pd
from copy import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings

def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df.reset_index(drop=True)

def smooth_columns(df, columns, window=3):
    df_copy = df.copy()
    for col in columns:
        df_copy[col] = df_copy[col].rolling(window=window, center=True, min_periods=1).mean()
    return df_copy

def clear(name):

    warnings.filterwarnings('ignore')

    data = pd.read_csv(name)

    numeric_cols = ['ID', 'Age', 'Income', 'WorkExperience', 'Satisfaction', 'SpendingHabits', 'Score1', 'Score2','Score3', 'Score4']
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

    data = data.dropna()

    data = data.reset_index(drop=True)

    data_second_press = remove_outliers_iqr(data, numeric_cols)

    data_smoothed = smooth_columns(data_second_press, numeric_cols)

    data_clean = data_smoothed.drop_duplicates().reset_index(drop=True)
    
    object_columns = data_clean.select_dtypes(include=['object']).columns
    labels = {}

    for col in object_columns:
        le = LabelEncoder()
        data_clean[col] = le.fit_transform(data_clean[col])
        labels[col] = le
    return data_clean