WITH order_items AS (
    SELECT * FROM {{ ref('dbt_stg_order_items') }}
),

products AS (
    SELECT * FROM {{ ref('dbt_stg_products') }}
),

categories AS (
    SELECT * FROM {{ ref('dbt_stg_category_translation') }}
),

sellers AS (
    SELECT * FROM {{ ref('dbt_stg_sellers') }}
)

SELECT
    oi.order_id,
    oi.order_item_id,
    oi.product_id,
    oi.seller_id,
    oi.price,
    oi.freight_value,
    p.product_category_name,
    s.seller_city,
    s.seller_state,
    INITCAP(COALESCE(
        REPLACE(c.product_category_name_english, '_', ' '),
        REPLACE(p.product_category_name, '_', ' ')
    )) AS product_category_name_english
FROM order_items AS oi
LEFT JOIN products AS p
    ON oi.product_id = p.product_id
LEFT JOIN categories AS c
    ON p.product_category_name = c.product_category_name
LEFT JOIN sellers AS s
    ON oi.seller_id = s.seller_id
