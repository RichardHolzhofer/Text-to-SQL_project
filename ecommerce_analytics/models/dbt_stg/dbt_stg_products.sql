WITH source AS (
    SELECT * FROM {{ source('dbt_stg', 'products') }}
),

renamed AS (
    SELECT
        product_id,
        _ingested_at AS ingested_at_utc,
        _file_name AS source_file,
        COALESCE(TRIM(LOWER(product_category_name)), 'unknown')
            AS product_category_name,
        COALESCE(CAST(product_name_length AS INTEGER), 0)
            AS product_name_length,
        COALESCE(CAST(description_length AS INTEGER), 0) AS description_length,
        COALESCE(CAST(product_photos_qty AS INTEGER), 0) AS product_photos_qty,
        COALESCE(CAST(product_weight_g AS INTEGER), 0) AS product_weight_g,
        COALESCE(CAST(product_lenght_cm AS INTEGER), 0) AS product_lenght_cm,
        COALESCE(CAST(product_height_cm AS INTEGER), 0) AS product_height_cm,
        COALESCE(CAST(product_width_cm AS INTEGER), 0) AS product_width_cm
    FROM source
)

SELECT
    product_id,
    product_category_name,
    product_name_length,
    description_length,
    product_photos_qty,
    product_weight_g,
    product_lenght_cm,
    product_height_cm,
    product_width_cm,
    ingested_at_utc,
    source_file
FROM renamed
