---
title: Advanced SQL using PostgreSQL
summary: SQL techniques including conditions, nested CASE statements, date handling with CURRENT_DATE, numeric functions like CEIL, FLOOR, and ROUND, string manipulation with SUBSTRING and CONCAT, and essential tools such as EXTRACT, CAST, and COALESCE.
date: 2025-11-28
authors:
  - admin
tags:
  - PostgreSQL
  - SQL
image:
  caption: ''
editable: false
# External Links (Buttons)
links:
  - name: Working with PostgreSQL
    url: blog/database-postgresql
---

SQL techniques including conditions, nested CASE statements, date handling with CURRENT_DATE, numeric functions like CEIL, FLOOR, and ROUND, string manipulation with SUBSTRING and CONCAT, and essential tools such as EXTRACT, CAST, and COALESCE.

### **Customers**

| customer_id | customer_name | region  | signup_date |
| ----------- | ------------- | ------- | ----------- |
| 101         | Anna 1        | West    | 2021-11-30  |
| 102         | Oliver 2      | Central | 2020-08-28  |
| 103         | Emma 3        | West    | 2021-04-16  |
| 104         | Anna 4        | Central | 2023-03-15  |
| 105         | Anna 5        | North   | 2020-10-26  |
| 106         | Mia 6         | Central | 2020-12-18  |
| 107         | David 7       | West    | 2021-09-03  |
| 108         | Chris 8       | East    | 2023-08-08  |
| 109         | Oliver 9      | South   | 2023-07-16  |
| 110         | John 10       | West    | 2022-12-21  |
| 111         | Anna 11       | West    | 2020-09-23  |
| 117         | Anna 17       | East    | 2023-04-18  |
| 120         | Chris 20      | Central | 2021-08-22  |
| 121         | Anna 21       | East    | 2023-05-24  |
| 122         | John 22       | West    | 2022-01-27  |
| 123         | Oliver 23     | South   | 2022-02-20  |
| 124         | Laura 24      | North   | 2021-07-09  |
| 125         | David 25      | East    | 2021-06-20  |
| 126         | John 26       | East    | 2023-06-30  |
| 127         | David 27      | South   | 2022-12-02  |
| 130         | Chris 30      | South   | 2021-04-14  |
| 131         | Anna 31       | North   | 2022-05-31  |
| 137         | Oliver 37     | North   | 2021-08-13  |
| 138         | David 38      | East    | 2021-06-12  |

---

### **Orders**


| order_id | customer_id | product_id | quantity | sale_date  | amount  |
| -------- | ----------- | ---------- | -------- | ---------- | ------- |
| 1        | 101         | 550        | 2        | 2021-06-13 | 753.18  |
| 2        | 120         | 528        | 1        | 2023-08-27 | 183.76  |
| 3        | 109         | 508        | 5        | 2023-04-09 | 781.4   |
| 4        | 109         | 512        | 3        | 2022-05-25 | 954.45  |
| 5        | 133         | 537        | 2        | 2022-04-07 | 849.3   |
| 6        | 137         | 538        | 5        | 2023-05-25 | 1781.9  |
| 7        | 124         | 540        | 4        | 2021-08-27 | 1771.6  |
| 8        | 110         | 536        | 5        | 2022-10-25 | 1858.2  |
| 9        | 138         | 542        | 4        | 2021-11-18 | 1142.44 |
| 10       | 127         | 511        | 5        | 2022-01-08 | 2281.3  |
| 11       | 101         | 542        | 4        | 2021-03-27 | 1142.44 |
| 12       | 108         | 509        | 1        | 2022-10-10 | 259.48  |
| 13       | 101         | 505        | 5        | 2022-09-23 | 826.05  |
| 14       | 117         | 516        | 4        | 2021-06-25 | 1853.64 |
| 15       | 122         | 523        | 2        | 2023-01-17 | 209.24  |


### **Products**

| product_id | product_name      | category    | price  |
| ---------- | ----------------- | ----------- | ------ |
| 505        | Vacuum Cleaner 5  | Home        | 165.21 |
| 506        | Laptop 6          | Kitchen     | 165.09 |
| 507        | Camera 7          | Home        | 419.05 |
| 508        | Novel 8           | Sports      | 156.28 |
| 509        | Headphones 9      | Books       | 259.48 |
| 510        | Vacuum Cleaner 10 | Home        | 130.88 |
| 511        | Smartwatch 11     | Electronics | 456.26 |
| 512        | Novel 12          | Kitchen     | 318.15 |
| 513        | Laptop 13         | Sports      | 426.4  |
| 514        | Mixer 14          | Sports      | 279.24 |
| 515        | Novel 15          | Kitchen     | 212.48 |
| 516        | Tennis Racket 16  | Sports      | 463.41 |
| 520        | Headphones 20     | Home        | 45.79  |
| 528        | Smartwatch 28     | Books       | 183.76 |
| 529        | Novel 29          | Books       | 413.2  |
| 530        | Novel 30          | Sports      | 42.82  |
| 536        | Blender 36        | Sports      | 371.64 |
| 537        | Vacuum Cleaner 37 | Electronics | 424.65 |
| 538        | Laptop 38         | Sports      | 356.38 |
| 539        | Smartwatch 39     | Electronics | 471.43 |
| 540        | Smartwatch 40     | Sports      | 442.9  |
| 541        | Smartwatch 41     | Sports      | 91.21  |
| 542        | Blender 42        | Sports      | 285.61 |
| 550        | Headphones 50     | Home        | 376.59 |


### 1. Show each order and calculate **how many days ago** it was placed.

```sql
SELECT 
    order_id,
    sale_date,
    CURRENT_DATE - sale_date AS days_ago
FROM orders;
``` 
### 2. Find customers who signed up more than **30 days ago**.

```sql
SELECT *
FROM customers
WHERE signup_date < CURRENT_DATE - INTERVAL '30 days';
``` 
### 3. Classify orders based on amount:

* Above 1000 → "High Value"
* Between 500–1000 → "Medium Value"
* Otherwise:
  * If quantity > 2 → "Low but Bulk"
  * Else → "Low Value"

```sql
SELECT 
    order_id,
    amount,
    quantity,
    CASE
        WHEN amount > 1000 THEN 'High Value'
        WHEN amount BETWEEN 500 AND 1000 THEN 'Medium Value'
        ELSE 
            CASE 
                WHEN quantity > 2 THEN 'Low but Bulk'
                ELSE 'Low Value'
            END
    END AS order_class
FROM orders;
``` 
### 4. Classify customers by region:

* North → "Prime Region"
* South or East → "Medium Region"
* Else:

  * If name starts with A–M → "Group 1"
  * Otherwise → "Group 2"

```sql
SELECT 
    customer_id,
    customer_name,
    region,
    CASE
        WHEN region = 'North' THEN 'Prime Region'
        WHEN region IN ('South', 'East') THEN 'Medium Region'
        ELSE 
            CASE 
                WHEN SUBSTRING(customer_name FROM 1 FOR 1) BETWEEN 'A' AND 'M'
                    THEN 'Group 1'
                ELSE 'Group 2'
            END
    END AS region_category
FROM customers;
``` 
### 5. Show order amount rounded, and also show it rounded up and down.

```sql
SELECT
    order_id,
    amount,
    CEIL(amount) AS amount_up,
    FLOOR(amount) AS amount_down,
    ROUND(amount, 0) AS amount_rounded
FROM orders;
``` 
### 6. Display only the first 3 letters of each customer’s region.

```sql
SELECT 
    customer_id,
    region,
    SUBSTRING(region FROM 1 FOR 3) AS short_region
FROM customers;
``` 
### 7. Extract first 2 letters of the product category.

```sql
SELECT 
    product_id,
    product_name,
    SUBSTRING(category FROM 1 FOR 2) AS cat_code
FROM products;
``` 
### 8. Display "CustomerName (CustomerID)" for each customer.

```sql
SELECT 
    CONCAT(customer_name, ' (', customer_id, ')') AS customer_label
FROM customers;
``` 
### 9. Create a formatted text: `Customer: <name> - Region: <region>`

```sql
SELECT 
    CONCAT('Customer: ', customer_name, ' - Region: ', region) AS info
FROM customers;
``` 
### 10. Show total sales per month.

```sql
SELECT 
    EXTRACT(MONTH FROM sale_date) AS month,
    SUM(amount) AS total_sales
FROM orders
GROUP BY month
ORDER BY month;
```
### 11. Show how many orders were created per **year**.

```sql
SELECT 
    EXTRACT(YEAR FROM sale_date) AS year,
    COUNT(*) AS orders_in_year
FROM orders
GROUP BY year;
```  
### 12. Convert sale_date to text and amount to integer.

```sql
SELECT 
    order_id,
    CAST(sale_date AS TEXT) AS date_text,
    CAST(amount AS INT) AS amount_integer
FROM orders;
``` 
### 13. If amount is NULL (missing), display 0 instead.

```sql
SELECT 
    order_id,
    COALESCE(amount, 0) AS fixed_amount
FROM orders;
``` 
### 14. Replace missing customer names with "No Name".

```sql
SELECT
    customer_id,
    COALESCE(customer_name, 'No Name') AS fixed_name
FROM customers;
``` 
### 15. Create a label: "Order #ID: $Amount"

```sql
SELECT
    CONCAT('Order #', order_id, ': $', CAST(amount AS TEXT)) AS order_label
FROM orders;
``` 
### 16. Show customer initials and region, using "Unknown" if name missing.

```sql
SELECT
    CONCAT(
        SUBSTRING(COALESCE(customer_name, 'Unknown') FROM 1 FOR 1),
        '.',
        region
    ) AS customer_initial_and_region
FROM customers;
``` 
### 17. CONCAT + SUBSTRING + CAST** Show a product label like:
`P<product_id> - <first 4 letters of name>`

```sql
SELECT 
    CONCAT(
        'P', CAST(product_id AS TEXT), ' - ', SUBSTRING(product_name FROM 1 FOR 4)
    ) AS product_label
FROM products;
``` 
### 18. Classify orders placed within the last 7 days as "Recent", within the same month as "This Month", otherwise "Old".

```sql
SELECT 
    order_id,
    sale_date,
    CASE 
        WHEN sale_date >= CURRENT_DATE - INTERVAL '7 days' THEN 'Recent'
        WHEN EXTRACT(MONTH FROM sale_date) = EXTRACT(MONTH FROM CURRENT_DATE)
             AND EXTRACT(YEAR FROM sale_date) = EXTRACT(YEAR FROM CURRENT_DATE)
             THEN 'This Month'
        ELSE 'Old'
    END AS order_age
FROM orders;
``` 
### 19. Label orders based on age:

* Less than 3 days → "Very Recent"
* Less than 10 days → "Recent"
* Otherwise → "Old"

```sql
SELECT 
    order_id,
    sale_date,
    CASE
        WHEN sale_date >= CURRENT_DATE - INTERVAL '3 days' THEN 'Very Recent'
        WHEN sale_date >= CURRENT_DATE - INTERVAL '10 days' THEN 'Recent'
        ELSE 'Old'
    END AS date_status
FROM orders;
``` 
### 20. ROUND + EXTRACT** Show average order amount per day of the week (rounded to 1 decimal).

```sql
SELECT 
    EXTRACT(DOW FROM sale_date) AS weekday,
    ROUND(AVG(amount), 1) AS avg_amount
FROM orders
GROUP BY weekday;
``` 
### 21. COALESCE + CEIL Use CEIL on amount but replace NULLs with 0.

```sql
SELECT 
    order_id,
    CEIL(COALESCE(amount, 0)) AS fixed_amount_up
FROM orders;
```

## Did you find this page helpful? Consider sharing it 🙌