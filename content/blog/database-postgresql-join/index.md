---
title: PostgreSQL Joins
summary: Querry using Joins.
date: 2025-12-03
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
  - name: Advanced SQL using PostgreSQL
    url: blog/database-postgresql-more

---

SQL joins are operations in a relational database that allow you to combine rows from two or more tables based on a related column between them. They are essential for querying data that is distributed across multiple tables in a structured manner.

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



### 1. Orders with customer name and product name

```sql
SELECT o.*, c.customer_name, p.product_name
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
INNER JOIN products p ON o.product_id = p.product_id;
```

###2. Orders placed by customers from the West region

```sql
SELECT o.*, c.customer_name
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
WHERE c.region = 'West';
```

###3. Product category, price, and order quantity

```sql
SELECT p.category, p.price, o.quantity
FROM orders o
INNER JOIN products p ON o.product_id = p.product_id;
```

###4. Orders placed in 2023

```sql
SELECT o.*, c.customer_name
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
WHERE EXTRACT(YEAR FROM o.sale_date) = 2023;
```

###5. Orders with Electronics products

```sql
SELECT o.*, p.product_name
FROM orders o
INNER JOIN products p ON o.product_id = p.product_id
WHERE p.category = 'Electronics';
```

###6. Customers whose signup date is before order date

```sql
SELECT o.*, c.customer_name, c.signup_date
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
WHERE c.signup_date < o.sale_date;
```

###7. Each customer and total money spent (quantity × price)

```sql
SELECT c.customer_id, c.customer_name,
       SUM(o.quantity * p.price) AS total_spent
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
INNER JOIN products p ON o.product_id = p.product_id
GROUP BY c.customer_id, c.customer_name;
```

###8. Products ordered more than 3 times

```sql
SELECT p.product_name, COUNT(o.order_id) AS total_orders
FROM orders o
INNER JOIN products p ON o.product_id = p.product_id
GROUP BY p.product_name
HAVING COUNT(o.order_id) > 3;
```

###9. Top 5 customers who spent the most money

```sql
SELECT c.customer_name,
       SUM(o.quantity * p.price) AS total_spent
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
INNER JOIN products p ON o.product_id = p.product_id
GROUP BY c.customer_name
ORDER BY total_spent DESC
LIMIT 5;
```

###10. Orders with total order value > 1000

```sql
SELECT o.*, p.product_name, (o.quantity * p.price) AS total_value
FROM orders o
INNER JOIN products p ON o.product_id = p.product_id
WHERE (o.quantity * p.price) > 1000;
```

###11. All customers and their orders (include customers with no orders)

```sql
SELECT c.*, o.order_id, o.sale_date
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id;
```

###12. Customers with no orders

```sql
SELECT c.*
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;
```

###13. Products and matching orders (include products never ordered)

```sql
SELECT p.*, o.order_id, o.quantity
FROM products p
LEFT JOIN orders o ON p.product_id = o.product_id;
```

###14. East region customers and their orders

```sql
SELECT c.*, o.order_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE c.region = 'East';
```

###15. All orders including those with invalid product_id

```sql
SELECT o.*, p.product_name
FROM orders o
LEFT JOIN products p ON o.product_id = p.product_id;
```

###16. All products and customers who bought them

```sql
SELECT p.*, o.order_id, c.customer_name
FROM products p
RIGHT JOIN orders o ON p.product_id = o.product_id
RIGHT JOIN customers c ON o.customer_id = c.customer_id;
```

###17. All customers and their orders (keep all orders)

```sql
SELECT o.*, c.customer_name
FROM orders o
RIGHT JOIN customers c ON o.customer_id = c.customer_id;
```

###18. Orders not matching any product

```sql
SELECT o.*
FROM orders o
LEFT JOIN products p ON o.product_id = p.product_id
WHERE p.product_id IS NULL;
```

###19. All customers and all orders (full outer join)

```sql
SELECT c.*, o.*
FROM customers c
FULL OUTER JOIN orders o
ON c.customer_id = o.customer_id;
```

###20. All products and all orders (full outer join)

```sql
SELECT p.*, o.*
FROM products p
FULL OUTER JOIN orders o
ON p.product_id = o.product_id;
```

###21. Unmatched customers + unmatched orders

```sql
SELECT *
FROM customers c
FULL OUTER JOIN orders o ON c.customer_id = o.customer_id
WHERE c.customer_id IS NULL OR o.customer_id IS NULL;
```

###22. All possible combinations of customers and products

```sql
SELECT c.customer_name, p.product_name
FROM customers c
CROSS JOIN products p;
```

###23. All combinations of products and regions

```sql
SELECT DISTINCT c.region, p.product_name
FROM customers c
CROSS JOIN products p;
```

###24. All customer–product combos where price > 400

```sql
SELECT c.customer_name, p.product_name, p.price
FROM customers c
CROSS JOIN products p
WHERE p.price > 400;
```

###25. Customers who share the same region (self join)

```sql
SELECT c1.customer_name AS customer1,
       c2.customer_name AS customer2,
       c1.region
FROM customers c1
INNER JOIN customers c2 
ON c1.region = c2.region AND c1.customer_id <> c2.customer_id;
```

###26. Customers with same signup date

```sql
SELECT c1.customer_name, c2.customer_name, c1.signup_date
FROM customers c1
INNER JOIN customers c2 
ON c1.signup_date = c2.signup_date 
AND c1.customer_id <> c2.customer_id;
```

###27. Same region but one signed up earlier

```sql
SELECT c1.customer_name AS earlier_signup,
       c2.customer_name AS later_signup,
       c1.region
FROM customers c1
INNER JOIN customers c2 
ON c1.region = c2.region 
AND c1.signup_date < c2.signup_date;
```

###28. Full order details (3-table join)

```sql
SELECT o.order_id, c.customer_name, p.product_name, p.category, p.price, o.quantity
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
INNER JOIN products p ON o.product_id = p.product_id;
```

###29. Customers + orders + total money spent per order

```sql
SELECT c.customer_name, o.order_id,
       (o.quantity * p.price) AS order_total
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
INNER JOIN products p ON o.product_id = p.product_id;
```

###30. Highest-value order in each product category

```sql
SELECT category, order_id, max(total_value)
FROM (
    SELECT p.category, o.order_id,
           (o.quantity * p.price) AS total_value
    FROM orders o
    INNER JOIN products p ON o.product_id = p.product_id
) t
GROUP BY category, order_id
ORDER BY category;
```



## Did you find this page helpful? Consider sharing it 🙌