SELECT 
    h.*, 
    s.client_id
FROM ga_hits h
JOIN ga_sessions s ON h.session_id = s.session_id