WITH payments AS (
    SELECT * FROM {{ ref('stg_olist__order_payments') }}
)

SELECT
    order_id,
    sum(payment_value) AS total_order_value,
    sum(CASE WHEN payment_type = 'credit_card' THEN payment_value ELSE 0 END)
        AS credit_card_amount,
    sum(CASE WHEN payment_type = 'voucher' THEN payment_value ELSE 0 END)
        AS voucher_amount,
    sum(CASE WHEN payment_type = 'boleto' THEN payment_value ELSE 0 END)
        AS boleto_amount,
    max(payment_installments) AS max_installments,
    count(payment_sequential) AS number_of_payment_methods,
    max(CASE WHEN payment_type = 'voucher' THEN 1 ELSE 0 END)
        AS has_voucher_payment
FROM payments
GROUP BY 1
