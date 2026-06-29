-- sql/features/default_aggregation.sql

WITH deduped_sessions AS (
    -- Шаг 1: Избавляемся от дубликатов сессий
    SELECT *
    FROM (
        SELECT s.*,
               ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY visit_date DESC, visit_time DESC) as rn
        FROM {raw_sessions_table} s
    ) ranked_sessions
    WHERE rn = 1
),

parsed_hits AS (
    -- Шаг 2: Базовая чистка дублей хитов (сбор таргета без влияния на фичи)
    SELECT 
        session_id,
        hit_number,
        hit_type,
        event_action,
        COALESCE(hit_time, 0) AS hit_time_clean,
        substring(hit_page_path from '/cars/(?:.*/)?([a-zA-Z0-9]+)(?:\?|$)') AS car_id,
        -- Маркер таргета (нужен ТОЛЬКО для финального event_value)
        CASE WHEN {target_actions_condition} THEN 1 ELSE 0 END AS is_target_hit
    FROM {raw_hits_table}
),

hits_features AS (
    -- Шаг 3: Честная агрегация. Считаем ТОЛЬКО базовую активность, исключая целевые действия
    SELECT 
        session_id,
        
        -- Базовые честные счетчики активности (минус таргет-клики)
        COUNT(CASE WHEN is_target_hit = 0 THEN 1 END) AS total_hits_count,
        COUNT(CASE WHEN hit_type = 'event' AND is_target_hit = 0 THEN 1 END) AS total_events_count,
        COUNT(DISTINCT CASE WHEN is_target_hit = 0 THEN event_action END) AS unique_event_actions,
        
        -- Честный интерес к автомобилям (сколько карточек посмотрел, без учета отправки заявок)
        COUNT(DISTINCT car_id) AS unique_cars_viewed,
        COUNT(car_id) AS total_car_views,
        
        -- Сбор целевой переменной (Y)
        MAX(is_target_hit) AS agg_target
    FROM parsed_hits
    GROUP BY session_id
)

-- Шаг 4: Финальный джойн контекста сессии и честных поведенческих фичей
SELECT 
    s.session_id,
    s.client_id,
    s.visit_date,
    s.visit_time,
    s.visit_number,
    s.utm_source,
    s.utm_medium,
    s.utm_campaign,
    s.utm_adcontent,
    s.device_category,
    s.device_os,
    s.device_brand,
    s.device_screen_resolution,
    s.device_browser,
    s.geo_country,
    s.geo_city,
    
    -- Честные поведенческие фичи
    COALESCE(h.total_hits_count, 0) AS total_hits_count,
    COALESCE(h.total_events_count, 0) AS total_events_count,
    COALESCE(h.unique_event_actions, 0) AS unique_event_actions,
    COALESCE(h.unique_cars_viewed, 0) AS unique_cars_viewed,
    COALESCE(h.total_car_views, 0) AS total_car_views,
    
    -- Доля просмотров авто от честных хитов сессии
    CASE 
        WHEN COALESCE(h.total_hits_count, 0) > 0 
        THEN ROUND(COALESCE(h.total_car_views, 0)::NUMERIC / h.total_hits_count, 4)
        ELSE 0 
    END AS car_view_ratio,
    
    -- Наш чистый вектор ответов (зависимая переменная)
    COALESCE(h.agg_target, 0) AS event_value

FROM deduped_sessions s
LEFT JOIN hits_features h ON s.session_id = h.session_id