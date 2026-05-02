WITH source AS (
    SELECT * FROM {{ source('olist', 'order_payments') }}
),

renamed AS (
    SELECT
        order_id,
        _ingested_at AS ingested_at_utc,
        _file_name AS source_file,
        CAST(payment_sequential AS INTEGER) AS payment_sequential,
        COALESCE(TRIM(LOWER(payment_type)), 'not_defined') AS payment_type,
        COALESCE(CAST(payment_installments AS INTEGER), 1)
            AS payment_installments,
        COALESCE(CAST(payment_value AS DECIMAL(10, 2)), 0) AS payment_value
    FROM source
)

SELECT
    order_id,
    payment_sequential,
    payment_type,
    payment_installments,
    payment_value,
    ingested_at_utc,
    source_file
FROM renamed
