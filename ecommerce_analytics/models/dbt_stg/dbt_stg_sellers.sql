WITH source AS (
    SELECT * FROM {{ source('dbt_stg', 'sellers') }}
),

renamed AS (
    SELECT
        seller_id,
        seller_zip_code_prefix AS seller_zip_code,
        _ingested_at AS ingested_at_utc,
        _file_name AS source_file,
        COALESCE(seller_city, 'unknown') AS seller_city,
        COALESCE(seller_state, 'unknown') AS seller_state
    FROM source
)

SELECT
    seller_id,
    seller_zip_code,
    seller_city,
    seller_state,
    ingested_at_utc,
    source_file
FROM renamed
