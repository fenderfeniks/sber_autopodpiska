-- sql/features/dirty_baseline_aggregation.sql
WITH deduped_sessions AS (
    -- Шаг 1: Избавляемся от дубликатов сессий
    SELECT *
    -- Выбираем все колонки из s, кроме технического номера строки
    FROM (
        SELECT s.*,
               ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY visit_date DESC, visit_time DESC) as rn
        FROM {raw_sessions_table} s
    ) ranked_sessions
    WHERE rn = 1
),
target_events AS (
    -- Шаг 2: Вытаскиваем только уникальные сессии, в которых случился таргет
    SELECT DISTINCT session_id
    FROM {raw_hits_table}
    WHERE {target_actions_condition}
)
-- Шаг 3: Джойним сессии с флагом таргета
SELECT 
    s.*,
    -- Если id сессии есть в таблице таргетов — ставим 1, если нет — 0
    CASE WHEN t.session_id IS NOT NULL THEN 1 ELSE 0 END AS event_value

FROM deduped_sessions s
LEFT JOIN target_events t ON s.session_id = t.session_id