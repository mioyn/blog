---
title: MAE and RMSE
date: '2025-12-06'
math: true
tags:
  - ML
  - Data Analytics
---
Common performance measures for regression problems are Mean Absolute Error (MAE), Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
# Mean Absolute Error - MAE

It is measured by taking the average of absolute difference(absolute error) between actula values and the predictions

It is thus an arithmetic average of the absolute errors
## Formula for MAE

$$
MAE = \frac{1}{n} \sum |y - \hat{y}|
$$

Where:
- $( y )$ Actual output value,
- $( \hat{y} )$ predicted output value,
- $( n )$ number of instance in the dataset.

# Root Mean Square Error - RMSE

It is measured by taking the square root of the average of the squared difference(error) between the prediction and actula value

It represents the sample standard deviation of the differences between predicted values and observed values(residuals).

## Formula for RMSE

$$
RMSE = \sqrt{ \frac{\sum (y - \hat{y})^2} {n} }
$$

### Note:
RMSE is more sensitive to outliers than MAE
