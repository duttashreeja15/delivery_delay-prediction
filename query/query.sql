
COPY (
WITH

-- 1. Deduplicate geolocation ZIP prefixes
geo AS (
    SELECT
        geolocation_zip_code_prefix AS zip_code,
        AVG(geolocation_lat) AS latitude,
        AVG(geolocation_lng) AS longitude
    FROM read_csv_auto(
        'data/olist_geolocation_dataset.csv'
    )
    WHERE geolocation_lat BETWEEN -35 AND 6
      AND geolocation_lng BETWEEN -75 AND -30
    GROUP BY geolocation_zip_code_prefix
),

-- 2. Aggregate item, product and seller features
--    BEFORE joining to orders or payments
items AS (
    SELECT
        i.order_id,

        COUNT(*) AS number_of_items,
        COUNT(DISTINCT i.product_id) AS unique_products,
        COUNT(DISTINCT i.seller_id) AS number_of_sellers,

        SUM(i.price) AS price,
        SUM(i.freight_value) AS freight_value,

        -- Convert grams to kilograms
        SUM(p.product_weight_g) / 1000.0
            AS total_product_weight_kg,

        AVG(p.product_weight_g) / 1000.0
            AS avg_product_weight_kg,

        AVG(p.product_length_cm) AS product_length_cm,
        AVG(p.product_height_cm) AS product_height_cm,
        AVG(p.product_width_cm) AS product_width_cm,

        -- Approximate total product volume in cm3
        SUM(
            p.product_length_cm
            * p.product_height_cm
            * p.product_width_cm
        ) AS total_product_volume_cm3,

        -- Representative seller/category for baseline
        ARG_MIN(
            s.seller_state,
            i.order_item_id
        ) AS seller_state,

        ARG_MIN(
            p.product_category_name,
            i.order_item_id
        ) AS product_category_name,

        AVG(sg.latitude) AS seller_latitude,
        AVG(sg.longitude) AS seller_longitude

    FROM read_csv_auto(
        'data/olist_order_items_dataset.csv'
    ) i

    LEFT JOIN read_csv_auto(
        'data/olist_products_dataset.csv'
    ) p
        ON i.product_id = p.product_id

    LEFT JOIN read_csv_auto(
        'data/olist_sellers_dataset.csv'
    ) s
        ON i.seller_id = s.seller_id

    LEFT JOIN geo sg
        ON s.seller_zip_code_prefix = sg.zip_code

    GROUP BY i.order_id
),

-- 3. Aggregate payments independently
payments AS (
    SELECT
        order_id,

        SUM(payment_value) AS payment_value,
        MAX(payment_installments) AS payment_installments,

        COUNT(*) AS number_of_payments,

        ARG_MAX(
            payment_type,
            payment_value
        ) AS payment_type

    FROM read_csv_auto(
        'data/olist_order_payments_dataset.csv'
    )
    GROUP BY order_id
),

-- 4. One record per order
orders_base AS (
    SELECT
        o.order_id,
        o.customer_id,
        o.order_status,

        o.order_purchase_timestamp,
        o.order_approved_at,
        o.order_delivered_carrier_date,
        o.order_delivered_customer_date,
        o.order_estimated_delivery_date,

        c.customer_state,
        c.customer_city,
        c.customer_zip_code_prefix,

        cg.latitude AS customer_latitude,
        cg.longitude AS customer_longitude,

        i.number_of_items,
        i.unique_products,
        i.number_of_sellers,

        i.price,
        i.freight_value,
        i.total_product_weight_kg,
        i.avg_product_weight_kg,
        i.product_length_cm,
        i.product_height_cm,
        i.product_width_cm,
        i.total_product_volume_cm3,

        i.seller_state,
        i.product_category_name,
        i.seller_latitude,
        i.seller_longitude,

        pay.payment_value,
        pay.payment_installments,
        pay.number_of_payments,
        pay.payment_type

    FROM read_csv_auto(
        'data/olist_orders_dataset.csv'
    ) o

    LEFT JOIN read_csv_auto(
        'data/olist_customers_dataset.csv'
    ) c
        ON o.customer_id = c.customer_id

    LEFT JOIN geo cg
        ON c.customer_zip_code_prefix = cg.zip_code

    LEFT JOIN items i
        ON o.order_id = i.order_id

    LEFT JOIN payments pay
        ON o.order_id = pay.order_id
)

-- 5. Final order-level dataset
SELECT
    *,

    -- Approximate great-circle distance (km)
    CASE
        WHEN seller_latitude IS NOT NULL
         AND seller_longitude IS NOT NULL
         AND customer_latitude IS NOT NULL
         AND customer_longitude IS NOT NULL
        THEN
            6371.0 * ACOS(
                LEAST(
                    1.0,
                    GREATEST(
                        -1.0,
                        SIN(RADIANS(customer_latitude))
                        * SIN(RADIANS(seller_latitude))
                        +
                        COS(RADIANS(customer_latitude))
                        * COS(RADIANS(seller_latitude))
                        * COS(
                            RADIANS(
                                customer_longitude
                                - seller_longitude
                            )
                        )
                    )
                )
            )
        ELSE NULL
    END AS shipping_distance_km

FROM orders_base

)
TO 'data/merged_orders.csv'
(FORMAT CSV, HEADER TRUE);
