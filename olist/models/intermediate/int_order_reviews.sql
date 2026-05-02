WITH reviews AS (
    SELECT * FROM {{ ref('stg_olist__order_reviews') }}
),

-- 1. Calculate the "Squashed" metrics
review_metrics AS (
    SELECT
        order_id,
        AVG(review_score) AS avg_review_score,
        COUNT(review_id) AS total_reviews_received
    FROM reviews
    GROUP BY 1
),

-- 2. Pick the "Best" single review to represent the text/dates
latest_review_content AS (
    SELECT
        order_id,
        review_comment_title,
        review_comment_message,
        review_creation_date,
        review_answer_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY order_id
            ORDER BY review_creation_date DESC, review_answer_timestamp DESC
        ) AS rn
    FROM reviews
)

SELECT
    m.order_id,
    m.avg_review_score,
    m.total_reviews_received,
    c.review_comment_title,
    c.review_comment_message,
    c.review_creation_date AS latest_review_creation_date,
    c.review_answer_timestamp AS latest_review_answer_timestamp
FROM review_metrics AS m
INNER JOIN latest_review_content AS c
    ON m.order_id = c.order_id
WHERE c.rn = 1
