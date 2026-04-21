WITH source AS (
    SELECT * FROM {{ source('dbt_stg', 'customers') }}
),

renamed AS (
    SELECT
        customer_id,
        customer_unique_id,
        _ingested_at AS ingested_at_utc,
        _file_name AS source_file,
        TRIM(customer_zip_code_prefix) AS customer_zip_code,
        COALESCE(TRIM(LOWER(customer_city)), 'unknown') AS customer_city,
        COALESCE(TRIM(LOWER(customer_state)), 'unknown') AS customer_state
    FROM source
)

SELECT
    customer_id,
    customer_unique_id,
    customer_zip_code,
    customer_city,
    customer_state,
    ingested_at_utc,
    source_file
FROM renamed
