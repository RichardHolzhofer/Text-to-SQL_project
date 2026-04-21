WITH source AS (
    SELECT * FROM {{ source('dbt_stg', 'geolocation') }}
),

renamed AS (
    SELECT
        geolocation_zip_code_prefix AS geolocation_zip_code,
        _ingested_at AS ingested_at_utc,
        _file_name AS source_file,
        CAST(geolocation_lat AS FLOAT) AS geolocation_lat,
        CAST(geolocation_lng AS FLOAT) AS geolocation_lng,
        COALESCE(geolocation_city, 'unknown') AS geolocation_city,
        COALESCE(geolocation_state, 'unknown') AS geolocation_state
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
