WITH orders AS (
    SELECT * FROM {{ ref('int_orders') }}
),

stg_customers AS (
    SELECT
        customer_id,
        customer_unique_id,
        customer_city,
        customer_state,
        customer_zip_code
    FROM {{ ref('stg_olist__customers') }}
)

SELECT
    o.order_id,
    c.customer_unique_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    o.total_order_value,
    o.credit_card_amount,
    o.voucher_amount,
    o.boleto_amount,
    o.has_voucher_payment,
    o.max_installments,
    o.avg_review_score AS review_score,
    o.latest_review_creation_date,
    o.latest_review_answer_timestamp,
    c.customer_city,
    c.customer_state,
    c.customer_zip_code,
    o.is_delivered_on_time,
    CASE
        WHEN o.order_status IN ('canceled', 'unavailable') THEN 0
        ELSE 1
    END AS is_valid_order,
    TRIM(
        CASE
            WHEN
                o.review_comment_title IS NOT NULL
                AND o.review_comment_message IS NOT NULL
                THEN
                    'Title: '
                    || o.review_comment_title
                    || ', Content: '
                    || o.review_comment_message
            WHEN o.review_comment_title IS NOT NULL
                THEN 'Title: ' || o.review_comment_title
            WHEN o.review_comment_message IS NOT NULL
                THEN 'Content: ' || o.review_comment_message
        END
    ) AS review_combined_text,
    CASE
        WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
            THEN
                DATEDIFF(
                    'day',
                    o.order_estimated_delivery_date,
                    o.order_delivered_customer_date
                )
        ELSE 0
    END AS days_delayed,
    DATEDIFF('day', o.order_purchase_timestamp, o.order_delivered_customer_date)
        AS actual_delivery_time_days,
    DATEDIFF('day', o.order_purchase_timestamp, o.order_estimated_delivery_date)
        AS estimated_delivery_time_days
FROM orders AS o
INNER JOIN stg_customers AS c ON o.customer_id = c.customer_id
