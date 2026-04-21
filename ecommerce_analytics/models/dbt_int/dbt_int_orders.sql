WITH orders AS (
    SELECT * FROM {{ ref('dbt_stg_orders') }}
),

payments AS (
    SELECT * FROM {{ ref('dbt_int_order_payments') }}
),

reviews AS (
    SELECT * FROM {{ ref('dbt_int_order_reviews') }}
)

SELECT
    o.order_id,
    o.customer_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    p.max_installments,
    r.avg_review_score,
    r.review_comment_title,
    r.review_comment_message,
    r.latest_review_creation_date,
    r.latest_review_answer_timestamp,
    COALESCE(p.total_order_value, 0) AS total_order_value,
    COALESCE(p.credit_card_amount, 0) AS credit_card_amount,
    COALESCE(p.voucher_amount, 0) AS voucher_amount,
    COALESCE(p.boleto_amount, 0) AS boleto_amount,
    COALESCE(p.has_voucher_payment, 0) AS has_voucher_payment,
    CASE
        WHEN o.order_delivered_customer_date <= o.order_estimated_delivery_date
            THEN 1
        ELSE 0
    END AS is_delivered_on_time
FROM orders AS o
LEFT JOIN payments AS p
    ON o.order_id = p.order_id
LEFT JOIN reviews AS r
    ON o.order_id = r.order_id
