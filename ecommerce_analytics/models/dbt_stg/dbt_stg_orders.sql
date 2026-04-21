WITH source AS (
    SELECT * FROM {{ source('dbt_stg', 'orders') }}
),

renamed AS (
    SELECT
        order_id,
        customer_id,
        _ingested_at AS ingested_at_utc,
        _file_name AS source_file,
        COALESCE(order_status, 'unknown') AS order_status,
        CAST(order_purchase_timestamp AS TIMESTAMP WITHOUT TIME ZONE)
            AS order_purchase_timestamp,
        CAST(order_approved_at AS TIMESTAMP WITHOUT TIME ZONE)
            AS order_approved_at,
        CAST(order_delivered_carrier_date AS TIMESTAMP WITHOUT TIME ZONE)
            AS order_delivered_carrier_date,
        CAST(order_delivered_customer_date AS TIMESTAMP WITHOUT TIME ZONE)
            AS order_delivered_customer_date,
        CAST(order_estimated_delivery_date AS TIMESTAMP WITHOUT TIME ZONE)
            AS order_estimated_delivery_date
    FROM source
)

SELECT
    order_id,
    customer_id,
    order_status,
    order_purchase_timestamp,
    order_approved_at,
    order_delivered_carrier_date,
    order_delivered_customer_date,
    order_estimated_delivery_date,
    ingested_at_utc,
    source_file
FROM renamed
