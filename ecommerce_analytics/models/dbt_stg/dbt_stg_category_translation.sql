WITH source AS (
    SELECT * FROM {{ ref('product_category_name_translation') }}
),

renamed AS (
    SELECT
        product_category_name,
        product_category_name_english
    FROM source
)

SELECT
    product_category_name,
    product_category_name_english
FROM renamed
