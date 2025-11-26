---
title: Working with PostgreSQL
summary: Some simple PostgreSQL to start learning SQL.
date: 2025-11-26
authors:
  - admin
tags:
  - PostgreSQL
  - SQL
image:
  caption: ''
---



### Logical Flow of SQL Query Execution in PostgreSQL
| Step   | Clause           | What happens                              |
| ------ | ---------------- | ----------------------------------------- |
| **1**  | `FROM`           | Selects the source tables and joins them. |
| **2**  | `ON`             | Applies join conditions (if joins exist). |
| **3**  | `JOIN`           | Combines rows from multiple tables.       |
| **4**  | `WHERE`          | Filters rows *before* grouping.           |
| **5**  | `GROUP BY`       | Groups rows into buckets.                 |
| **6**  | `HAVING`         | Filters groups created by GROUP BY.       |
| **7**  | `SELECT`         | Selects columns and expressions.          |
| **8**  | `DISTINCT`       | Removes duplicates (after SELECT).        |
| **9**  | `ORDER BY`       | Sorts the results.                        |
| **10** | `LIMIT / OFFSET` | Returns only a subset of rows.            |


### Drop and create table
```sql
DROP TABLE IF EXISTS public.sales_data;

CREATE TABLE sales_data (
	sale_id INT,
	customer_id VARCHAR(10),
	product VARCHAR(50),
	region VARCHAR(10),
	amount NUMERIC(10, 2),
	sale_date DATE
);
```
### Data to execute below sql

```javascript
sale_id,customer_id,product,region,amount,sale_date
101,C001,Laptop,North,900,2024-01-01
102,C002,Phone,South,450,2024-01-02
103,C003,Tablet,East,700,2024-01-03
104,C004,Headphones,West,250,2024-01-04
105,C005,Camera,North,1200,2024-01-05
106,C001,Laptop,South,500,2024-01-06
107,C002,Phone,East,800,2024-01-07
108,C003,Tablet,West,650,2024-01-08
109,C004,Headphones,North,1100,2024-01-09
110,C005,Camera,South,400,2024-01-10
111,C001,Laptop,East,950,2024-01-11
112,C002,Phone,West,350,2024-01-12
113,C003,Tablet,North,600,2024-01-13
114,C004,Headphones,South,1000,2024-01-14
115,C005,Camera,East,700,2024-01-15
116,C001,Laptop,West,300,2024-01-16
117,C002,Phone,North,750,2024-01-17
118,C003,Tablet,South,1250,2024-01-18
119,C004,Headphones,East,400,2024-01-19
120,C005,Camera,West,850,2024-01-20
```

### 1. Show all records from the sales_data table.

```sql
SELECT * FROM sales_data;
```

### 2. Retrieve the first 10 rows from the sales_data table.

```sql
SELECT * FROM sales_data LIMIT 10;
```

### 3. List all unique regions where sales occurred.

```sql
SELECT DISTINCT region FROM sales_data;
```

### 4. List all distinct products sold.

```sql
SELECT DISTINCT product FROM sales_data;
```

### 5. Show all sales made in the North region.

```sql
SELECT * FROM sales_data WHERE region = 'North';
```

### 6. Retrieve all sales where the amount is greater than 500.

```sql
SELECT * FROM sales_data WHERE amount > 500;
```

### 7. Find all sales that occurred on January 1, 2024.

```sql
SELECT * FROM sales_data WHERE sale_date = '2024-01-01';
```

### 8. Find laptop sales made in the South region.

```sql
SELECT * FROM sales_data WHERE region = 'South' AND product = 'Laptop';
```

### 9. Show all sales made in either the East or West region.

```sql
SELECT * FROM sales_data WHERE region = 'East' OR region = 'West';
```

### 10. Retrieve all sales except those where the product is Mobile.

```sql
SELECT * FROM sales_data WHERE product != 'Mobile';
```

### 11. Display all sales sorted by amount from highest to lowest.

```sql
SELECT * FROM sales_data ORDER BY amount DESC;
```

### 12. Show the earliest 10 sales based on sale date.

```sql
SELECT * FROM sales_data ORDER BY sale_date LIMIT 10;
```

### 13. Retrieve all sales where the product name starts with the letter ‘P’.

```sql
SELECT * FROM sales_data WHERE product LIKE 'P%';
```

### 14. Show all sales where the region name contains the letter ‘a’.

```sql
SELECT * FROM sales_data WHERE region LIKE '%a%';
```

### 15. Retrieve all sales made in North or East regions.

```sql
SELECT * FROM sales_data WHERE region IN ('North', 'East');
```

### 16. Find all sales where the amount is between 200 and 800.

```sql
SELECT * FROM sales_data WHERE amount BETWEEN 200 AND 800;
```

### 17. Retrieve sales made between January 1 and January 15, 2024.

```sql
SELECT * FROM sales_data WHERE sale_date BETWEEN '2024-01-01' AND '2024-01-15';
```

### 18. Calculate the total sales amount.

```sql
SELECT SUM(amount) FROM sales_data;
```

### 19. Find the average sale amount.

```sql
SELECT AVG(amount) FROM sales_data;
```

### 20. Count the total number of sales records.

```sql
SELECT COUNT(*) FROM sales_data;
```

### 21. How many distinct products were sold?

```sql
SELECT COUNT(DISTINCT product) FROM sales_data;
```

### 22. What is the highest sales amount?

```sql
SELECT MAX(amount) FROM sales_data;
```

### 23. What is the lowest sales amount?

```sql
SELECT MIN(amount) FROM sales_data;
```

### 24. Calculate total sales amount for each region.

```sql
SELECT region, SUM(amount) FROM sales_data GROUP BY region;
```

### 25. Count how many times each product was sold.

```sql
SELECT product, COUNT(product) FROM sales_data GROUP BY product;
```

### 26. Find the average sales amount for each region.

```sql
SELECT region, AVG(amount) AS avg_amt FROM sales_data GROUP BY region;
```

### 27. Show regions whose total sales exceed 3000.

```sql
SELECT region, SUM(amount) AS sal_amt
FROM sales_data
GROUP BY region
having SUM(amount) > 3000;
```

### 28. List products that were sold more than 2 times.

```sql
SELECT product, COUNT(product)
FROM sales_data
GROUP BY product
HAVING COUNT(product) > 2;
```

### 29. Find all North region sales where the product starts with ‘C’, sorted by amount descending.

```sql
SELECT * FROM sales_data
WHERE region = 'North' AND product LIKE 'C%'
ORDER BY amount DESC;
```

### 30. Retrieve sales over 500 from East or West regions.

```sql
SELECT * FROM sales_data
WHERE amount > 500 AND region in ('East', 'West');
```

### 31. Find all Phone sales between January 1 and January 10, 2024.

```sql
SELECT * FROM sales_data
WHERE product = 'Phone'
AND sale_date BETWEEN '2024-01-01' AND '2024-01-10';
```

### 32. Find the top 3 customers by total purchase amount.

```sql
SELECT customer_id, SUM(amount) AS tot_sum
FROM sales_data
GROUP BY customer_id
ORDER BY tot_sum desc
LIMIT 3;
```

### 33. Show total and average sales amount for each region.

```sql
SELECT SUM(amount) AS tot_sl, AVG(amount) AS avg_sal, region
FROM sales_data
GROUP BY region;
```

### 34. Calculate total sales for each month.

```sql
SELECT EXTRACT(MONTH FROM sale_date) AS mon, SUM(amount)
FROM sales_data
GROUP BY mon;
```

### 35. Calculate daily total sales by sale date.

```sql
SELECT EXTRACT(day FROM sale_date) AS dy, SUM(amount)
FROM sales_data
GROUP BY sale_date;
```

### Try it.
Find Tablet or Camera sales over 800
Sales between two dates and not in the North
Sales for Camera in the region West
Customers who bought more than one product type
Highest 5 amounts
Latest 3 sales
Total sales per product
Average sales per customer
Highest sale per region


## Did you find this page helpful? Consider sharing it 🙌
