WITH source AS (
    SELECT * FROM {{ source('olist', 'geolocation') }}
),

renamed AS (
    SELECT
        _ingested_at AS ingested_at_utc,
        _file_name AS source_file,
        TRIM(geolocation_zip_code_prefix) AS geolocation_zip_code,
        CAST(geolocation_lat AS FLOAT) AS geolocation_lat,
        CAST(geolocation_lng AS FLOAT) AS geolocation_lng,
        COALESCE(TRIM(LOWER(geolocation_city)), 'unknown') AS geolocation_city,
        COALESCE(TRIM(LOWER(geolocation_state)), 'unknown') AS geolocation_state
    FROM source
)

SELECT
    geolocation_zip_code,
    geolocation_lat,
    geolocation_lng,
    geolocation_city,
    geolocation_state,
    ingested_at_utc,
    source_file
FROM renamed
