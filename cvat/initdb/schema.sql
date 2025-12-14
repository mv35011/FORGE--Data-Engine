-- Drop tables to ensure clean slate
DROP TABLE IF EXISTS audits;
DROP TABLE IF EXISTS qc_flags;
DROP TABLE IF EXISTS golden_annotations;
DROP TABLE IF EXISTS annotations;
DROP TABLE IF EXISTS tasks;
DROP TABLE IF EXISTS projects;

CREATE TABLE IF NOT EXISTS projects (
    project_id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES projects(project_id),
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50),
    assignee VARCHAR(255),
    retrieved_at TIMESTAMP WITH TIME ZONE,
    qc_status VARCHAR(50) DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS annotations (
    annotation_id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES tasks(task_id),
    keyframe_name VARCHAR(255) NOT NULL,
    person_id INTEGER NOT NULL,
    xtl REAL, ytl REAL, xbr REAL, ybr REAL,
    attributes JSONB
);

CREATE TABLE IF NOT EXISTS golden_annotations (
    golden_annotation_id SERIAL PRIMARY KEY,
    original_project_id INTEGER,
    original_task_id_A INTEGER,
    original_task_id_B INTEGER,
    keyframe_name VARCHAR(255) NOT NULL,
    person_id INTEGER NOT NULL,
    xtl REAL, ytl REAL, xbr REAL, ybr REAL,
    attributes JSONB,
    adjudicated_by VARCHAR(255),
    adjudicated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(keyframe_name, person_id)
);

-- UPDATED AUDITS TABLE
CREATE TABLE IF NOT EXISTS audits (
    audit_id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES tasks(task_id), -- Added this column
    keyframe_name VARCHAR(255) NOT NULL,
    person_id INTEGER NOT NULL,
    original_consensus_attributes JSONB,
    auditor_attributes JSONB,
    is_overturn BOOLEAN,
    auditor_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS qc_flags (
    flag_id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES tasks(task_id),
    keyframe_name VARCHAR(255),
    issue_type VARCHAR(50),
    severity VARCHAR(20),
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
