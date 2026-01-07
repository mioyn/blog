---
title: 'ML Regression California housing model'
date: '2025-12-13'
math: true
tags:
  - ML
  - Data Analytics
---


```python
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
```
## Load Data
```python
def load_housingdata():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.exists():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        print("Downloaded housing dataset.")
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
            print("Extracted housing dataset.")
    housing_csv_path = Path("datasets/housing/housing.csv")
    return pd.read_csv(housing_csv_path)

housing_df = load_housingdata()
housing_df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>

```python
housing_df.info()
housing_df.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
housing_df['ocean_proximity'].value_counts()
```

    ocean_proximity
    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: count, dtype: int64

```python
# housing_df.hist(bins=50, figsize=(20,15))
# plt.show()
numeric_columns_df = housing_df.select_dtypes(include=[np.number])
numeric_columns = numeric_columns_df.columns

for i, column in enumerate(numeric_columns, 1):
    sns.histplot(data=housing_df[column], bins=50, kde=True)
    plt.show()
```

    
![png](output_5_0.png)
    

    
![png](output_5_1.png)
    

    
![png](output_5_2.png)
    

    
![png](output_5_3.png)
    

    
![png](output_5_4.png)
    

    
![png](output_5_5.png)
    

    
![png](output_5_6.png)
    

    
![png](output_5_7.png)
    

    
![png](output_5_8.png)
    
## Binning/Discretization

Grouping continuous data into discrete bins or intervals.
```python
housing_df['income_cat'] = pd.cut(
    housing_df['median_income'],
    bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

plt.subplots( 1, 1, figsize=(12, 6))
housing_df['income_cat'].hist()
plt.show()
```

    
![png](output_6_0.png)
    
## Create Test and Train Set
```python
strat_train_set, strat_test_set = train_test_split(
    housing_df,
    test_size=0.2,
    stratify=housing_df['income_cat'],
    random_state=42
)

strat_train_set["income_cat"].value_counts() / len(strat_train_set)
```

    income_cat
    3    0.350594
    2    0.318859
    4    0.176296
    5    0.114462
    1    0.039789
    Name: count, dtype: float64
### Drop income_cat once Test and Train Set are created
```python
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```
## Visualizing geographical Data
```python
housing_c = strat_train_set.copy()
housing_c.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2, grid=True)
plt.show()
```

    
![png](output_9_0.png)
    

```python
housing_c.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                s=housing_c["population"]/100, label="population",
                c="median_house_value", cmap=plt.get_cmap("turbo"), colorbar=True,
                legend=True, sharex=False, figsize=(12,8))
plt.show()
```

    
![png](output_10_0.png)
    
## Look for Corelations
```python
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

sns.pairplot(housing_c[attributes])

# Display the plot
plt.show()
```

    
![png](output_11_0.png)
    

```python
corr_matrix_numeric_columns = housing_c[attributes].corr()

plt.figure(figsize=(8, 6))

# 4. Plot the heatmap using Seaborn
sns.heatmap(corr_matrix_numeric_columns, 
            annot=True,      # Annotate cells with correlation values
            cmap='coolwarm', # Choose a color map (e.g., 'coolwarm', 'viridis')
            fmt=".2f",       # Format annotations to 2 decimal places
            linewidths=0.5   # Add lines to divide cells
           )

plt.title("Correlation Heatmap")
plt.show()
```

    
![png](output_12_0.png)
    
## Prepare Data
### Seperate predictors and labels 
```python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```
### Clean data

## Custom Tranformer
```python
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import OneHotEncoder

from sklearn.cluster import KMeans
```

```python
# using kmeans to find cluster similarity
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, random_state=None, gamma=1.0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.gamma = gamma
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, input_features=None):
        return [f"cluster {i} similarity" for i in range(self.n_clusters)]
```
### Pipelines
```python
# function to find ratio for columns 
# number of bedrooms  "total_bedrooms" / "total_rooms"
def column_ratio(x):
    return x[:, [0]] / x[:, [1]]

# add colum name for created ratio column
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

# ratio pipeline
def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())
# logtransform for heavy tailed distribution
logging_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())

# One hot encoder for categorical data
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

cluster_simil = ClusterSimilarity(n_clusters=10, random_state=42, gamma=1.0)

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler())
```

```python
preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_household", ratio_pipeline(), ["population", "households"]),
    ("log", logging_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
    ("geo", cluster_simil, ["longitude", "latitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object))
], remainder=default_num_pipeline)
```

```python
housing_prepared = preprocessing.fit_transform(housing)
housing_prepared.shape
```

    (16512, 24)

```python
preprocessing.get_feature_names_out()
```

    array(['bedrooms__ratio', 'rooms_per_house__ratio',
           'people_per_household__ratio', 'log__total_bedrooms',
           'log__total_rooms', 'log__population', 'log__households',
           'log__median_income', 'geo__cluster 0 similarity',
           'geo__cluster 1 similarity', 'geo__cluster 2 similarity',
           'geo__cluster 3 similarity', 'geo__cluster 4 similarity',
           'geo__cluster 5 similarity', 'geo__cluster 6 similarity',
           'geo__cluster 7 similarity', 'geo__cluster 8 similarity',
           'geo__cluster 9 similarity', 'cat__ocean_proximity_<1H OCEAN',
           'cat__ocean_proximity_INLAND', 'cat__ocean_proximity_ISLAND',
           'cat__ocean_proximity_NEAR BAY', 'cat__ocean_proximity_NEAR OCEAN',
           'remainder__housing_median_age'], dtype=object)

## Find Best modal
### LinearRegression
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lin_reg = make_pipeline(
    preprocessing,
    LinearRegression()
)
lin_reg.fit(housing, housing_labels)
housing_predictions = lin_reg.predict(housing)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
print("RMSE:", np.sqrt(lin_mse))

mean_avg_error = np.mean(np.abs(housing_predictions - housing_labels))
print("MAE:", mean_avg_error)
```

    RMSE: 68972.88910758452
    MAE: 51301.0044458925

### DecisionTreeRegressor
```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(
    preprocessing,
    DecisionTreeRegressor(random_state=42)
)
tree_reg.fit(housing, housing_labels)
housing_predictions_tree = tree_reg.predict(housing)
tree_mse = mean_squared_error(housing_labels, housing_predictions_tree)
print("RMSE:", np.sqrt(tree_mse))
mean_avg_error_tree = np.mean(np.abs(housing_predictions_tree - housing_labels))
print("MAE:", mean_avg_error_tree)
```

    RMSE: 0.0
    MAE: 0.0

```python
from sklearn.model_selection import cross_val_score
scores = -cross_val_score(
    tree_reg,
    housing,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

print("Scores:", scores)
print("Mean:", scores.mean())
```

    Scores: [64607.89604624 66409.0627187  66203.41031283 65863.76434319
     68086.79141027 66534.53308312 66923.49892937 68532.21066423
     66367.48227837 66208.69621463]
    Mean: 66573.73460009348

### RandomForestRegressor
```python
from sklearn.ensemble import RandomForestRegressor
forest_reg = make_pipeline(
    preprocessing,
    RandomForestRegressor(random_state=42)
)
forest_rmse_scores = -cross_val_score(
    forest_reg,
    housing,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
print("Scores:", forest_rmse_scores)
print("Mean:", forest_rmse_scores.mean())
```

    Scores: [46576.85817312 47238.61187815 45495.97664925 46488.27129295
     45911.17628527 47730.60806253 47561.15494003 49140.83221016
     47121.06435179 47116.37414628]
    Mean: 47038.09279895258

### GridSearchCV and RandomizedSearchCV

```python
from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42))
])
param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
    'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
    'random_forest__max_features': [6, 8, 10]},
]

grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)
```

```python
grid_search.best_params_
```

    {'preprocessing__geo__n_clusters': 15, 'random_forest__max_features': 6}

```python
cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_preprocessing__geo__n_clusters</th>
      <th>param_random_forest__max_features</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>10.292507</td>
      <td>0.965225</td>
      <td>0.164285</td>
      <td>0.002938</td>
      <td>15</td>
      <td>6</td>
      <td>{'preprocessing__geo__n_clusters': 15, 'random...</td>
      <td>-42725.423800</td>
      <td>-43708.197434</td>
      <td>-44334.935606</td>
      <td>-43589.518946</td>
      <td>662.417543</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14.025353</td>
      <td>1.996004</td>
      <td>0.226480</td>
      <td>0.040759</td>
      <td>15</td>
      <td>8</td>
      <td>{'preprocessing__geo__n_clusters': 15, 'random...</td>
      <td>-43486.175916</td>
      <td>-43819.842374</td>
      <td>-44899.968680</td>
      <td>-44068.662323</td>
      <td>603.399271</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8.105849</td>
      <td>0.142929</td>
      <td>0.231984</td>
      <td>0.001934</td>
      <td>10</td>
      <td>4</td>
      <td>{'preprocessing__geo__n_clusters': 10, 'random...</td>
      <td>-43797.854175</td>
      <td>-44036.240246</td>
      <td>-44960.694004</td>
      <td>-44264.929475</td>
      <td>501.513170</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>14.398225</td>
      <td>3.974331</td>
      <td>0.264121</td>
      <td>0.083238</td>
      <td>10</td>
      <td>6</td>
      <td>{'preprocessing__geo__n_clusters': 10, 'random...</td>
      <td>-43709.661050</td>
      <td>-44163.463178</td>
      <td>-44966.539107</td>
      <td>-44279.887778</td>
      <td>519.680433</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11.885352</td>
      <td>0.523488</td>
      <td>0.240539</td>
      <td>0.025509</td>
      <td>10</td>
      <td>6</td>
      <td>{'preprocessing__geo__n_clusters': 10, 'random...</td>
      <td>-43709.661050</td>
      <td>-44163.463178</td>
      <td>-44966.539107</td>
      <td>-44279.887778</td>
      <td>519.680433</td>
      <td>4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>17.811009</td>
      <td>1.847737</td>
      <td>0.212763</td>
      <td>0.020285</td>
      <td>15</td>
      <td>10</td>
      <td>{'preprocessing__geo__n_clusters': 15, 'random...</td>
      <td>-44350.273180</td>
      <td>-44485.169004</td>
      <td>-45211.211272</td>
      <td>-44682.217819</td>
      <td>378.087094</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.616109</td>
      <td>0.284040</td>
      <td>0.245921</td>
      <td>0.008608</td>
      <td>8</td>
      <td>4</td>
      <td>{'preprocessing__geo__n_clusters': 8, 'random_...</td>
      <td>-44511.046790</td>
      <td>-44519.243029</td>
      <td>-45214.107933</td>
      <td>-44748.132584</td>
      <td>329.511319</td>
      <td>7</td>
    </tr>
    <tr>
      <th>10</th>
      <td>13.476036</td>
      <td>0.891266</td>
      <td>0.199532</td>
      <td>0.043493</td>
      <td>10</td>
      <td>8</td>
      <td>{'preprocessing__geo__n_clusters': 10, 'random...</td>
      <td>-44498.988402</td>
      <td>-44883.300454</td>
      <td>-45264.655671</td>
      <td>-44882.314842</td>
      <td>312.583131</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>16.691502</td>
      <td>1.605912</td>
      <td>0.321681</td>
      <td>0.101431</td>
      <td>10</td>
      <td>8</td>
      <td>{'preprocessing__geo__n_clusters': 10, 'random...</td>
      <td>-44498.988402</td>
      <td>-44883.300454</td>
      <td>-45264.655671</td>
      <td>-44882.314842</td>
      <td>312.583131</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11.113520</td>
      <td>0.259273</td>
      <td>0.222916</td>
      <td>0.006448</td>
      <td>8</td>
      <td>6</td>
      <td>{'preprocessing__geo__n_clusters': 8, 'random_...</td>
      <td>-44851.823543</td>
      <td>-44827.316247</td>
      <td>-45654.572385</td>
      <td>-45111.237392</td>
      <td>384.326110</td>
      <td>10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13.695136</td>
      <td>0.884320</td>
      <td>0.171400</td>
      <td>0.013169</td>
      <td>10</td>
      <td>10</td>
      <td>{'preprocessing__geo__n_clusters': 10, 'random...</td>
      <td>-45335.811283</td>
      <td>-45498.761729</td>
      <td>-45851.568040</td>
      <td>-45562.047017</td>
      <td>215.259578</td>
      <td>11</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14.699613</td>
      <td>0.363115</td>
      <td>0.240795</td>
      <td>0.006014</td>
      <td>8</td>
      <td>8</td>
      <td>{'preprocessing__geo__n_clusters': 8, 'random_...</td>
      <td>-45567.827559</td>
      <td>-45323.269449</td>
      <td>-46133.989451</td>
      <td>-45675.028820</td>
      <td>339.544610</td>
      <td>12</td>
    </tr>
    <tr>
      <th>0</th>
      <td>7.429751</td>
      <td>0.124911</td>
      <td>0.239206</td>
      <td>0.020718</td>
      <td>5</td>
      <td>4</td>
      <td>{'preprocessing__geo__n_clusters': 5, 'random_...</td>
      <td>-46226.404919</td>
      <td>-46146.401866</td>
      <td>-47195.120579</td>
      <td>-46522.642455</td>
      <td>476.634202</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.035088</td>
      <td>0.236341</td>
      <td>0.261770</td>
      <td>0.024467</td>
      <td>5</td>
      <td>6</td>
      <td>{'preprocessing__geo__n_clusters': 5, 'random_...</td>
      <td>-46473.214688</td>
      <td>-46383.628085</td>
      <td>-47620.692379</td>
      <td>-46825.845050</td>
      <td>563.230649</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.286089</td>
      <td>0.584856</td>
      <td>0.235794</td>
      <td>0.017432</td>
      <td>5</td>
      <td>8</td>
      <td>{'preprocessing__geo__n_clusters': 5, 'random_...</td>
      <td>-46839.745574</td>
      <td>-46904.579633</td>
      <td>-47813.212568</td>
      <td>-47185.845925</td>
      <td>444.404127</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'preprocessing__geo__n_clusters': randint(low=3, high=50),
    'random_forest__max_features': randint(low=2, high=20)}

rnd_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions=param_distribs,
    n_iter=10,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42
)
rnd_search.fit(housing, housing_labels)
rnd_search.best_params_
```

    {'preprocessing__geo__n_clusters': 45, 'random_forest__max_features': 9}

### Find important features
```python
final_model = rnd_search.best_estimator_
feature_importances = final_model.named_steps['random_forest'].feature_importances_
feature_names = final_model.named_steps['preprocessing'].get_feature_names_out()
sorted(zip(feature_importances, feature_names), reverse=True)
```

    [(np.float64(0.18599734460509473), 'log__median_income'),
     (np.float64(0.07338850855844488), 'cat__ocean_proximity_INLAND'),
     (np.float64(0.06556941990883974), 'bedrooms__ratio'),
     (np.float64(0.05364871007672531), 'rooms_per_house__ratio'),
     (np.float64(0.04598870861894748), 'people_per_household__ratio'),
     (np.float64(0.04175269214442518), 'geo__cluster 30 similarity'),
     (np.float64(0.025976797232869678), 'geo__cluster 25 similarity'),
     (np.float64(0.023595895886342252), 'geo__cluster 36 similarity'),
     (np.float64(0.02021056221732893), 'geo__cluster 9 similarity'),
     (np.float64(0.018606917076661445), 'geo__cluster 34 similarity'),
     (np.float64(0.01813798837462886), 'geo__cluster 37 similarity'),
     (np.float64(0.017404353166326745), 'geo__cluster 18 similarity'),
     (np.float64(0.01677838614384489), 'geo__cluster 1 similarity'),
     (np.float64(0.015459009666188978), 'geo__cluster 7 similarity'),
     (np.float64(0.015325731028175922), 'geo__cluster 32 similarity'),
     (np.float64(0.015073772015038346), 'geo__cluster 13 similarity'),
     (np.float64(0.014272160962173803), 'geo__cluster 35 similarity'),
     (np.float64(0.014180636461860477), 'geo__cluster 0 similarity'),
     (np.float64(0.01374636449823899), 'geo__cluster 3 similarity'),
     (np.float64(0.013572305708469519), 'geo__cluster 28 similarity'),
     (np.float64(0.012940349694228718), 'geo__cluster 26 similarity'),
     (np.float64(0.012738123746761943), 'geo__cluster 31 similarity'),
     (np.float64(0.011654237215152623), 'geo__cluster 19 similarity'),
     (np.float64(0.011628003598059721), 'geo__cluster 6 similarity'),
     (np.float64(0.011134113333125396), 'geo__cluster 24 similarity'),
     (np.float64(0.011042979326385047), 'remainder__housing_median_age'),
     (np.float64(0.010907388443940416), 'geo__cluster 43 similarity'),
     (np.float64(0.010847192663592164), 'geo__cluster 44 similarity'),
     (np.float64(0.010592244492858262), 'geo__cluster 10 similarity'),
     (np.float64(0.010512467290844922), 'geo__cluster 23 similarity'),
     (np.float64(0.010458665615386447), 'geo__cluster 41 similarity'),
     (np.float64(0.010261910692851671), 'geo__cluster 40 similarity'),
     (np.float64(0.00975730698309749), 'geo__cluster 2 similarity'),
     (np.float64(0.009659933222114479), 'geo__cluster 12 similarity'),
     (np.float64(0.009574969190852867), 'geo__cluster 14 similarity'),
     (np.float64(0.008199144719918424), 'geo__cluster 20 similarity'),
     (np.float64(0.008141941480860804), 'geo__cluster 33 similarity'),
     (np.float64(0.00759676121996469), 'geo__cluster 8 similarity'),
     (np.float64(0.007576298012849029), 'geo__cluster 22 similarity'),
     (np.float64(0.007346290789504318), 'geo__cluster 39 similarity'),
     (np.float64(0.006898774333063982), 'geo__cluster 4 similarity'),
     (np.float64(0.006794731845079839), 'log__total_rooms'),
     (np.float64(0.006514889773323567), 'log__population'),
     (np.float64(0.006350528211987124), 'geo__cluster 27 similarity'),
     (np.float64(0.0063375587499023365), 'geo__cluster 16 similarity'),
     (np.float64(0.006231053672395538), 'geo__cluster 38 similarity'),
     (np.float64(0.006121348345871485), 'log__households'),
     (np.float64(0.00584984200158211), 'log__total_bedrooms'),
     (np.float64(0.005678310466685012), 'geo__cluster 15 similarity'),
     (np.float64(0.005479729990673466), 'geo__cluster 29 similarity'),
     (np.float64(0.005348325088535127), 'geo__cluster 42 similarity'),
     (np.float64(0.004866251452445484), 'geo__cluster 17 similarity'),
     (np.float64(0.004495340541933026), 'geo__cluster 11 similarity'),
     (np.float64(0.004418821635620684), 'geo__cluster 5 similarity'),
     (np.float64(0.003534473250529128), 'geo__cluster 21 similarity'),
     (np.float64(0.0018324246573418507), 'cat__ocean_proximity_<1H OCEAN'),
     (np.float64(0.0015282226447271793), 'cat__ocean_proximity_NEAR OCEAN'),
     (np.float64(0.00043259703422473606), 'cat__ocean_proximity_NEAR BAY'),
     (np.float64(3.0190221102670295e-05), 'cat__ocean_proximity_ISLAND')]

```python
from sklearn.metrics import root_mean_squared_error

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
final_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)
print("Final RMSE on Test Set:", final_rmse)
```

    Final RMSE on Test Set: 41445.533268606625

## Save Model
```python
import joblib
joblib.dump(final_model, "housing_model.pkl")
```

    ['housing_model.pkl']

