-- Удаляем таблицы, если они уже существуют (для безопасного перезапуска)
DROP TABLE IF EXISTS ga_hits;
DROP TABLE IF EXISTS ga_sessions;

-- 1. Создаем таблицу действий (хитов)
CREATE TABLE ga_hits (
    session_id VARCHAR(255),
    hit_date DATE,
    hit_time TIME,
    hit_number INTEGER,
    hit_type VARCHAR(50),
    hit_referer TEXT,
    hit_page_path TEXT,
    event_category VARCHAR(100),
    event_action VARCHAR(100),
    event_label TEXT,
    event_value NUMERIC
);

-- 2. Создаем таблицу сессий со всеми атрибутами
CREATE TABLE ga_sessions (
    session_id VARCHAR(255),
    client_id VARCHAR(255),
    visit_date DATE,
    visit_time TIME,
    visit_number INTEGER,
    utm_source VARCHAR(100),
    utm_medium VARCHAR(100),
    utm_campaign VARCHAR(100),
    utm_keyword VARCHAR(255),
    device_category VARCHAR(50),
    device_os VARCHAR(50),
    device_brand VARCHAR(100),
    device_model VARCHAR(100),
    device_screen_resolution VARCHAR(50),
    device_browser VARCHAR(100),
    geo_country VARCHAR(100),
    geo_city VARCHAR(100)
);

-- 3. Создаем индексы по session_id. 
-- Они критически важны, так как объединять таблицы (JOIN) мы будем именно по этому полю.
CREATE INDEX idx_hits_session ON ga_hits(session_id);
CREATE INDEX idx_sessions_session ON ga_sessions(session_id);