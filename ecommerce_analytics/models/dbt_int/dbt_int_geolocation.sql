WITH geolocation AS (
    SELECT * FROM {{ ref('dbt_stg_geolocation') }}
),

ranked_names AS (
    SELECT
        geolocation_zip_code,
        geolocation_city,
        geolocation_state,
        ROW_NUMBER() OVER (
            PARTITION BY geolocation_zip_code
            ORDER BY COUNT(*) DESC, geolocation_city ASC
        ) AS rn
    FROM geolocation
    GROUP BY 1, 2, 3
)

SELECT
    g.geolocation_zip_code,
    n.geolocation_city,
    n.geolocation_state,
    AVG(g.geolocation_lat) AS avg_lat,
    AVG(g.geolocation_lng) AS avg_lng
FROM geolocation AS g
INNER JOIN ranked_names AS n
    ON g.geolocation_zip_code = n.geolocation_zip_code
WHERE n.rn = 1
GROUP BY 1, 2, 3
