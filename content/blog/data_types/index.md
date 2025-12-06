---
title: 'Types of Data'
date: '2025-12-06'
math: true
tags:
  - Probability
  - Data Analytics
---


### **1. Discrete Data**  
- **Counted values** (whole numbers). Consists of distinct, separate values that can be counted.
- Always whole numbers (no decimals or fractions)
- **Examples:** Number of students, cars, calls.  

### **2. Continuous Data**  
- **Measured values** (can be fractional).  
- **Examples:** Height, temperature, weight.  

### **3. Nominal Data**  
- Categorical, and consists of labels or names without any order.
- Cannot be ranked or compared meaningfully.
- Used for classification purposes.
- **Examples:** Eye color, pet type, brand names.  

### **4. Ordinal Data**  
- Categorical but has a meaningful order or ranking.
- Can be ranked, but the difference between categories is not necessarily equal.
- Used in surveys, ratings, and classifications.  
- **Examples:** Movie ratings, education level, survey responses.  

However, you can count how many times each category appears. For example:

You can count how many people have blue eyes $(nominal)$.

You can count how many students rated a movie as 5-star $(ordinal)$.

You can count how many times a temperature measurement falls within a certain range $(continuous)$.

So, while the individual values of continuous, nominal, and ordinal data are not inherently countable, the frequency of occurrences within these types can be counted

# Classifying these:

1. **Ranking of cities by quality of life** → Berlin (1st), Zurich (2nd), Vienna (3rd)  
2. **Blood types of patients** → A, B, AB, O  
3. **Sprint times of athletes** → 9.58s, 10.12s, 10.37s  
4. **Number of successful penalty kicks** → 5, 7, 9, 12  
5. **Movie ratings** → ⭐⭐⭐⭐, ⭐⭐⭐, ⭐⭐⭐⭐⭐  

Answer

1. Ordinal  
2. Nominal  
3. Continuous  
4. Discrete  
5. Ordinal  

# Classifying these:
1. **Monthly income of employees in a company (in euros)** → €2,500, €3,200, €4,100, €5,500
2. **Types of smartphones owned by a group of people** → iPhone, Samsung, Google Pixel, OnePlus
3. **Student grades in an exam** → A, B, C, D, F
4. **Number of books read by each person in a year** → 2, 5, 8, 12, 15
5. **Temperature recorded over a day (in Celsius)** → 14.5°C, 18.2°C, 22.8°C, 20.1°C

Answer

1. Discrete
2. Nominal
3. Ordinal
4. Discrete
5. Continuous

A single dataset can contain multiple types of data. Many real-world datasets mix discrete, continuous, nominal, and ordinal variables.

Examples of Mixed Data Types in Datasets
1. Customer Demographics Dataset (e.g., a retail database)
  - Discrete: Number of purchases made per month
  - Continuous: Customer's annual income
  - Nominal: Gender (Male, Female, Non-binary)
  - Ordinal: Customer satisfaction rating (Poor, Average, Good, Excellent)

2. Medical Records Dataset (e.g., hospital patient data)
  - Discrete: Number of doctor visits per year
  - Continuous: Patient’s weight and height
  - Nominal: Blood type (A, B, AB, O)
  - Ordinal: Severity of illness (Mild, Moderate, Severe)

3. Housing Dataset (e.g., California Housing Data)
  - Discrete: Number of bedrooms in a house
  - Continuous: House price or square footage
  - Nominal: Type of house (apartment, condo, villa)
  - Ordinal: Condition rating (Poor, Fair, Good, Excellent)

Real-world data is diverse—different aspects of information are needed for meaningful analysis.

Combining multiple data types helps in predictive modeling (e.g., predicting house prices based on categorical and numerical variables).

Machine learning models handle mixed data types differently (some need numerical encoding for categorical features).
