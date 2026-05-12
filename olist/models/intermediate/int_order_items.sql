WITH order_items AS (
    SELECT * FROM {{ ref('stg_olist__order_items') }}
),

products AS (
    SELECT * FROM {{ ref('stg_olist__products') }}
),

categories AS (
    SELECT * FROM {{ ref('stg_olist__category_translations') }}
),

sellers AS (
    SELECT * FROM {{ ref('stg_olist__sellers') }}
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
    MD5(CONCAT(
        COALESCE(CAST(oi.order_id AS STRING), '_null_'),
        COALESCE(CAST(oi.order_item_id AS STRING), '_null_')
    )) AS order_item_pk,
    LOWER(COALESCE(
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
