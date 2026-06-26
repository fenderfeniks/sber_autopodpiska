-- sql/features/default_aggregation.sql
WITH deduped_sessions AS (
    -- Шаг 1: Избавляемся от дубликатов сессий, если они есть в сырой таблице
    SELECT *
    FROM (
        SELECT s.*,
               ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY visit_date DESC, visit_time DESC) as rn
        FROM {raw_sessions_table} s
    ) ranked_sessions
    WHERE rn = 1
),
parsed_hits AS (
    -- Шаг 2: Чистим и парсим хиты (дубли по hit_number внутри сессии отсекаем)
    SELECT *
    FROM (
        SELECT 
            session_id,
            hit_number,
            hit_type,
            event_action,
            COALESCE(hit_time, 0) AS hit_time_clean,
            substring(hit_page_path from '/cars/(?:.*/)?([a-zA-Z0-9]+)(?:\?|$)') AS car_id,
            CASE WHEN {target_actions_condition} THEN 1 ELSE 0 END AS is_target_hit,
            ROW_NUMBER() OVER (PARTITION BY session_id, hit_number ORDER BY hit_time ASC) as rn_hit
        FROM {raw_hits_table}
    ) ranked_hits
    WHERE rn_hit = 1
),
hits_features AS (
    -- Шаг 3: Агрегируем чистые хиты до уровня сессии
    SELECT 
        session_id,
        COUNT(hit_number) AS total_hits_count,
        MIN(hit_time_clean) AS first_hit_time_ms,
        MAX(hit_time_clean) AS last_hit_time_ms,
        
        -- Фичи вовлеченности
        COUNT(CASE WHEN hit_type = 'event' THEN 1 END) AS total_events_count,
        COUNT(DISTINCT event_action) AS unique_event_actions,
        
        -- Фичи по машинам
        COUNT(DISTINCT CASE WHEN car_id IS NOT NULL THEN car_id END) AS unique_cars_viewed,
        COUNT(CASE WHEN car_id IS NOT NULL THEN 1 END) AS total_car_views,
        
        -- Был ли первый хит просмотром машины (1 - да, 0 - нет)
        MAX(CASE WHEN hit_number = 1 AND car_id IS NOT NULL THEN 1 ELSE 0 END) AS is_first_hit_car_view,
        
        -- Собираем таргет на сессию
        MAX(is_target_hit) AS agg_target,
        -- Находим номер первого хита, где случился таргет
        MIN(CASE WHEN is_target_hit = 1 THEN hit_number END) AS first_target_hit_number
    FROM parsed_hits
    GROUP BY session_id
)
-- Шаг 4: Джойним дедуплицированные сессии и агрегированные хиты
SELECT 
    s.*,
    COALESCE(h.total_hits_count, 0) AS total_hits_count,
    COALESCE(h.first_hit_time_ms, 0) AS first_hit_time_ms,
    COALESCE(h.last_hit_time_ms, 0) AS last_hit_time_ms,
    COALESCE(h.total_events_count, 0) AS total_events_count,
    COALESCE(h.unique_event_actions, 0) AS unique_event_actions,
    COALESCE(h.unique_cars_viewed, 0) AS unique_cars_viewed,
    COALESCE(h.total_car_views, 0) AS total_car_views,
    COALESCE(h.is_first_hit_car_view, 0) AS is_first_hit_car_view,
    
    COALESCE(h.agg_target, 0) AS event_value, 
    
    CASE 
        WHEN h.first_target_hit_number IS NOT NULL THEN h.first_target_hit_number - 1
        ELSE COALESCE(h.total_hits_count, 0)
    END AS hits_before_target,
    
    -- Доля просмотров авто от всех хитов сессии
    CASE 
        WHEN COALESCE(h.total_hits_count, 0) > 0 
        THEN ROUND(COALESCE(h.total_car_views, 0)::NUMERIC / h.total_hits_count, 4)
        ELSE 0 
    END AS car_view_ratio

FROM deduped_sessions s -- Используем чистые сессии вместо исходной таблицы
LEFT JOIN hits_features h ON s.session_id = h.session_id