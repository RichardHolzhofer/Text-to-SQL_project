WITH source AS (
    SELECT * FROM {{ source('dbt_stg', 'order_reviews') }}
),

renamed AS (
    SELECT
        review_id,
        order_id,
        review_comment_title,
        review_comment_message,
        _ingested_at AS ingested_at_utc,
        _file_name AS source_file,
        CAST(review_score AS INTEGER) AS review_score,
        CAST(review_creation_date AS TIMESTAMP WITHOUT TIME ZONE)
            AS review_creation_date,
        CAST(review_answer_timestamp AS TIMESTAMP WITHOUT TIME ZONE)
            AS review_answer_timestamp
    FROM source
)

SELECT
    review_id,
    order_id,
    review_score,
    review_comment_title,
    review_comment_message,
    review_creation_date,
    review_answer_timestamp,
    ingested_at_utc,
    source_file
FROM renamed
