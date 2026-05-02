WITH customers AS (
    SELECT * FROM {{ ref('int_customers') }}
),

orders AS (
    SELECT * FROM {{ ref('int_orders') }}
),

stg_customers AS (
    SELECT
        customer_id,
        customer_unique_id
    FROM {{ ref('stg_olist__customers') }}
),

order_mapping AS (
    SELECT
        o.order_id,
        c.customer_unique_id,
        o.total_order_value,
        o.order_status
    FROM orders AS o
    INNER JOIN stg_customers AS c ON o.customer_id = c.customer_id
),

spending_stats AS (
    SELECT
        customer_unique_id,
        SUM(
            CASE
                WHEN
                    order_status NOT IN ('canceled', 'unavailable')
                    THEN total_order_value
                ELSE 0
            END
        ) AS lifetime_value,
        AVG(
            CASE
                WHEN
                    order_status NOT IN ('canceled', 'unavailable')
                    THEN total_order_value
            END
        ) AS average_order_value
    FROM order_mapping
    GROUP BY 1
)

SELECT
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    c.customer_zip_code,
    c.total_valid_orders,
    c.total_canceled_orders,
    c.first_purchase_at,
    c.last_purchase_at,
    c.is_returning_customer,
    COALESCE(s.lifetime_value, 0) AS lifetime_value,
    COALESCE(s.average_order_value, 0) AS average_order_value,
    DATEDIFF('day', c.last_purchase_at, CURRENT_TIMESTAMP()) AS recency_days
FROM customers AS c
LEFT JOIN spending_stats AS s ON c.customer_unique_id = s.customer_unique_id
