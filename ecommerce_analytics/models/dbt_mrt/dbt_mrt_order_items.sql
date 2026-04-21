WITH order_items AS (
    SELECT * FROM {{ ref('dbt_int_order_items') }}
),

orders AS (
    SELECT
        order_id,
        order_status,
        order_purchase_timestamp
    FROM {{ ref('dbt_int_orders') }}
)

SELECT
    oi.order_id,
    oi.order_item_id,
    oi.product_id,
    oi.seller_id,
    oi.price,
    oi.freight_value,
    oi.product_category_name,
    oi.product_category_name_english,
    oi.seller_city,
    oi.seller_state,
    o.order_status,
    o.order_purchase_timestamp
FROM order_items AS oi
INNER JOIN orders AS o ON oi.order_id = o.order_id
