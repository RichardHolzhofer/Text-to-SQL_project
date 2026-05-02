WITH customers AS (
    SELECT * FROM {{ ref('stg_olist__customers') }}
),

orders AS (
    SELECT * FROM {{ ref('stg_olist__orders') }}
),

customer_stats AS (
    SELECT
        c.customer_unique_id,
        count(
            CASE
                WHEN
                    o.order_status NOT IN ('canceled', 'unavailable')
                    THEN o.order_id
            END
        ) AS total_valid_orders,
        count(CASE WHEN o.order_status = 'canceled' THEN o.order_id END)
            AS total_canceled_orders,
        min(o.order_purchase_timestamp) AS first_purchase_at,
        max(o.order_purchase_timestamp) AS last_purchase_at
    FROM customers AS c
    INNER JOIN orders AS o ON c.customer_id = o.customer_id
    GROUP BY 1
),

latest_location AS (
    SELECT
        c.customer_unique_id,
        c.customer_city,
        c.customer_state,
        c.customer_zip_code,
        row_number() OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY
                CASE
                    WHEN o.order_status = 'delivered' THEN 1
                    WHEN o.order_status = 'unknown' THEN 3
                    ELSE 2
                END,
                o.order_purchase_timestamp DESC,
                o.order_id DESC
        ) AS rn
    FROM customers AS c
    INNER JOIN orders AS o ON c.customer_id = o.customer_id
)

SELECT
    s.customer_unique_id,
    l.customer_city,
    l.customer_state,
    l.customer_zip_code,
    s.total_valid_orders,
    s.total_canceled_orders,
    s.first_purchase_at,
    s.last_purchase_at,
    (s.total_valid_orders > 1) AS is_returning_customer
FROM customer_stats AS s
INNER JOIN latest_location AS l ON s.customer_unique_id = l.customer_unique_id
WHERE l.rn = 1
