CREATE TABLE IF NOT EXISTS model_capabilities (
  model_id TEXT PRIMARY KEY,
  capabilities_json TEXT NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS capability_overrides (
  model_id TEXT PRIMARY KEY,
  overrides_json TEXT NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ground_truth_capabilities (
  model_id TEXT PRIMARY KEY,
  capabilities_json TEXT NOT NULL,
  source TEXT DEFAULT 'runtime_detection',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
