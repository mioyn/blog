---
title: Feature Engineering
date: '2025-12-07'
math: true
tags:
  - ML
  - Data Analytics
---

## Imputation (Handling Missing Values)
Many real world datasets contain missing values, often encoded as blanks, NaNs or other placeholders. A basic strategy to use incomplete datasets is to discard entire rows and/or columns containing missing values. However, this comes at the price of losing data which may be valuable (even though incomplete). A better strategy is to impute the missing values.

Set the missing values to some value(Zero, the mean, the median, etc.)

```
# Fill missing values using forward fill then backward fill
df[required_cols] = df[required_cols].fillna(method='ffill').fillna(method='bfill')
```
Another way is to use Scikit-learn class:SimpleImputer

```python
import numpy as np
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy='mean') # median, most_frequent
imp.fit([[1, 2], [np.nan, 3], [7, 6]])

X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))
```

    [[4.         2.        ]
     [6.         3.66666667]
     [7.         6.        ]]

## Outlier Detection & Removal
### Interquartile Range (IQR) Method

```python
# Function to remove outliers using IQR
def remove_outliers_iqr(df, column):
  #Compute Q1 (25th percentile) and Q3 (75th percentile).
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)

  # calculate IQR
  IQR = Q3 - Q1

  # define lower and upper bounds - values outside is considerd as outliers
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  
  # remove outliers
  df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
  return df_cleaned

# Remove outliers for column salary
df = remove_outliers_iqr(df.copy(), 'salary')
```

## Encoding
Scikit-Learn provides a OneHotEncoder class to convert categorical values into one-hot vectors
Output of OneHotEncoder is sparse matrix. A sparse matrix contains mostly zeros. Internally it only stores the nonzero values and their positions. when you have more categories it will save plenty of memory and speed up computations.

```python
from sklearn.preprocessing import OneHotEncoder

housing_cat = housing_df[['ocean_proximity']]
encoder = OneHotEncoder()
cat_1hot = encoder.fit_transform(housing_cat)

# get list of categories
encoder.categories_
```

## Feature Scaling

The MinMax Scaler is a data preprocessing technique used to transform numerical features in a dataset to a specific range, typically between 0 and 1. This is also commonly referred to as normalization. The core idea is to make the minimum value in a feature column equal to 0, the maximum value equal to 1, and all other values fall proportionally in between.

This is achieved using a simple mathematical formula applied to each data point:

  $X_{\text{scaled}}=\frac{X-X_{\text{min}}}{X_{\text{max}}-X_{\text{min}}}$

Where:
  
  * $X$ is the original data point.
  * $X_{\text{min}}$ is the minimum value for that feature in the dataset.
  * $X_{\text{max}}$ is the maximum value for that feature in the dataset.

```python
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_scaled = min_max_scaler.fit_transform(housing_num_df)
```

###Real-Time Predictive Maintenance
In a factory, sensors continuously monitor the condition of critical machinery (like a turbine) to predict mechanical failures before they happen.

#### The Raw Data:
Imagine the data stream includes three features collected every second:
  * **Vibration Level:** Measured in meters per second squared $(m/s^2)$, typically a small range (e.g., 0.1 to 5.0).
  * **Temperature:** Measured in degrees Celsius $(^{\circ }C)$, with a range of operating temperatures $(e.g., 20{}^{\circ }C - 200{}^{\circ }C)$.
  * **Energy Consumption:** Measured in Megawatt-hours (MWh), a large range (e.g., 10 MWh to 500 MWh).

####The Problem Without Scaling:
An algorithm that relies on distance metrics (like K-Nearest Neighbors to detect an anomaly) would be completely dominated by the Energy Consumption feature because its values (hundreds) are much larger than the Vibration Level values (single digits) or Temperature values (tens/hundreds). The model would essentially ignore minor, but critical, changes in vibration that signal an impending failure.

#### Applying a Scaler (e.g., MinMax Scaler):
To ensure all features contribute equally, a MinMaxScaler is applied: Vibration Level is scaled from its range of [0.1, 5.0] to [0, 1].Temperature is scaled from its range of [20, 200] to [0, 1].Energy Consumption is scaled from its range of [10, 500] to [0, 1].
#### The Result:
The scaled features are all on the same playing field. The machine learning model can now effectively analyze patterns across all three features simultaneously. A sudden spike in vibration, even though its raw value is small, will have an impact comparable to a spike in temperature or energy consumption in the scaled data, leading to faster and more accurate predictions of potential equipment failure.

## Binning/Discretization

Grouping continuous data into discrete bins or intervals.

```python
housing_df['income_cat'] = pd.cut(
    housing_df['income'],
    bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
).astype(int)

plt.subplots( 1, 1, figsize=(12, 6))
housing_df['income_cat'].hist()
plt.show()
```
![hist](incom_cat_hist.png "income_cat histogram")

The code successfully converts a continuous, numerical variable (**income**) into a categorical, ordinal variable (**income_cat**).

This is often done to perform stratified sampling ensuring that a machine learning model sees a representative sample of all income levels during training, which can be crucial for building a fair and accurate predictive model.
