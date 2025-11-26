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
select * from sales_data;
```

### 2. Retrieve the first 10 rows from the sales_data table.

```sql
select * from sales_data LIMIT 10;
```

### 3. List all unique regions where sales occurred.

```sql
select DISTINCT region from sales_data;
```

### 4. List all distinct products sold.

```sql
select DISTINCT product from sales_data;
```

### 5. Show all sales made in the North region.

```sql
select * from sales_data where region = 'North';
```

### 6. Retrieve all sales where the amount is greater than 500.

```sql
select * from sales_data where amount > 500;
```

### 7. Find all sales that occurred on January 1, 2024.

```sql
select * from sales_data where sale_date = '2024-01-01';
```

### 8. Find laptop sales made in the South region.

```sql
select * from sales_data where region = 'South' AND product = 'Laptop';
```

### 9. Show all sales made in either the East or West region.

```sql
select * from sales_data where region = 'East' OR region = 'West';
```

### 10. Retrieve all sales except those where the product is Mobile.

```sql
select * from sales_data where product != 'Mobile';
```

### 11. Display all sales sorted by amount from highest to lowest.

```sql
select * from sales_data order by amount desc;
```

### 12. Show the earliest 10 sales based on sale date.

```sql
select * from sales_data order by sale_date limit 10;
```

### 13. Retrieve all sales where the product name starts with the letter ‘P’.

```sql
select * from sales_data where product like 'P%';
```

### 14. Show all sales where the region name contains the letter ‘a’.

```sql
select * from sales_data where region like '%a%';
```

### 15. Retrieve all sales made in North or East regions.

```sql
select * from sales_data where region in ('North', 'East');
```

### 16. Find all sales where the amount is between 200 and 800.

```sql
select * from sales_data where amount between 200 and 800;
```

### 17. Retrieve sales made between January 1 and January 15, 2024.

```sql
select * from sales_data where sale_date between '2024-01-01' and '2024-01-15';
```

### 18. Calculate the total sales amount.

```sql
select sum(amount) from sales_data;
```

### 19. Find the average sale amount.

```sql
select avg(amount) from sales_data;
```

### 20. Count the total number of sales records.

```sql
select count(*) from sales_data;
```

### 21. How many distinct products were sold?

```sql
select count(DISTINCT product) from sales_data;
```

### 22. What is the highest sales amount?

```sql
select max(amount) from sales_data;
```

### 23. What is the lowest sales amount?

```sql
select min(amount) from sales_data;
```

### 24. Calculate total sales amount for each region.

```sql
select region, SUM(amount) from sales_data group by region;
```

### 25. Count how many times each product was sold.

```sql
select product, count(product) from sales_data group by product;
```

### 26. Find the average sales amount for each region.

```sql
select region, avg(amount) as avg_amt from sales_data group by region;
```

### 27. Show regions whose total sales exceed 3000.

```sql
select region, sum(amount) as sal_amt
from sales_data
group by region
having sum(amount) > 3000;
```

### 28. List products that were sold more than 2 times.

```sql
select product, count(product)
from sales_data
group by product
having count(product) > 2;
```

### 29. Find all North region sales where the product starts with ‘C’, sorted by amount descending.

```sql
select * from sales_data
where region = 'North' and product like 'C%'
order by amount desc;
```

### 30. Retrieve sales over 500 from East or West regions.

```sql
select * from sales_data
where amount > 500 and region in ('East', 'West');
```

### 31. Find all Phone sales between January 1 and January 10, 2024.

```sql
select * from sales_data
where product = 'Phone'
and sale_date between '2024-01-01' and '2024-01-10';
```

### 32. Find the top 3 customers by total purchase amount.

```sql
select customer_id, sum(amount) as tot_sum
from sales_data
group by customer_id
order by tot_sum desc
limit 3;
```

### 33. Show total and average sales amount for each region.

```sql
select sum(amount) as tot_sl, AVG(amount) as avg_sal, region
from sales_data
group by region;
```

### 34. Calculate total sales for each month.

```sql
select EXTRACT(MONTH from sale_date) as mon, sum(amount)
from sales_data
group by mon;
```

### 35. Calculate daily total sales by sale date.

```sql
select EXTRACT(day from sale_date) as dy, sum(amount)
from sales_data
group by sale_date;
```

## Did you find this page helpful? Consider sharing it 🙌
