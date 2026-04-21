WITH reviews AS (
    SELECT * FROM {{ ref('dbt_int_order_reviews') }}
)

SELECT
    order_id,
    avg_review_score AS review_score,
    total_reviews_received,
    review_comment_title,
    review_comment_message,
    latest_review_creation_date,
    latest_review_answer_timestamp,
    TRIM(
        CASE
            WHEN
                review_comment_title IS NOT NULL
                AND review_comment_message IS NOT NULL
                THEN
                    'Title: '
                    || review_comment_title
                    || ', Content: '
                    || review_comment_message
            WHEN review_comment_title IS NOT NULL
                THEN 'Title: ' || review_comment_title
            WHEN review_comment_message IS NOT NULL
                THEN 'Content: ' || review_comment_message
        END
    ) AS review_combined_text,
    LENGTH(review_comment_message) AS review_comment_length
FROM reviews
