WITH source AS (
    SELECT * FROM {{ source('dbt_stg', 'order_items') }}
),

renamed AS (
    SELECT
        order_id,
        product_id,
        seller_id,
        _ingested_at AS ingested_at_utc,
        _file_name AS source_file,
        CAST(order_item_id AS INTEGER) AS order_item_id,
        CAST(shipping_limit_date AS TIMESTAMP WITHOUT TIME ZONE)
            AS shipping_limit_date,
        COALESCE(CAST(price AS DECIMAL(10, 2)), 0) AS price,
        COALESCE(CAST(freight_value AS DECIMAL(10, 2)), 0) AS freight_value
    FROM source
)

SELECT
    order_id,
    order_item_id,
    product_id,
    seller_id,
    shipping_limit_date,
    price,
    freight_value,
    ingested_at_utc,
    source_file
FROM renamed
