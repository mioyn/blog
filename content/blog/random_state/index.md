---
title: Random State
date: '2025-12-06'
math: true
tags:
  - ML
---


Random_state acts as a seed for the random number generator used to shuffle and split the data. Assigning a specific value to random_state ensures that the split is reproducible. In other words, every time you run the script with the same seed, you get the same train-test split. This is crucial for debugging and sharing results with others.

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
Number you choose for random_state doesn't really matter; it's an arbitrary number fed into the program to dictate that the program should use a specific seed of randomness

## Example in NumPy

```python
import numpy as np

np.random.seed(42)
print(np.random.randint(0, 100, 5))
```

    [51 92 14 71 60]

A seed is just an integer that initializes the random number generator's internal state.

Computers don't generate true randomness; they use pseudo-random number generators $(PRNGs)$.

A PRNG is an algorithm that takes a starting value $(seed)$ and applies a mathematical recurrence relation to produce the next number.

Each "***next random number***" is a deterministic function of the previous state.

That's why random sequences are reproducible when you set the seed.

## Stratify
It is important to have a sufficient number of instances in the dataset for each stratum, or else the estimate of the stratum's importance may be biased. this means that you should not have too many strata and each stratum should be large enough.

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
