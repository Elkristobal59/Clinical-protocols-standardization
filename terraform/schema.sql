-- Activation de l'extension pgvector si elle n'existe pas
CREATE EXTENSION IF NOT EXISTS vector;

-- Création de la table principale
CREATE TABLE IF NOT EXISTS clinical_trials_data_biobert (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(255) NOT NULL,
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
    raw_text TEXT NOT NULL,
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Création de l'index HNSW pour optimiser la recherche par similarité cosinus
CREATE INDEX IF NOT EXISTS clinical_trials_biobert_embedding_idx 
ON clinical_trials_data_biobert 
USING hnsw (embedding vector_cosine_ops);

-- ==============================================================
-- 🚀 ARCHITECTURE AUTOMATISEE : VIDAGE AUTO (Data Lifecycle)
-- ==============================================================
-- 1. Activation de l'extension de tâches planifiées (Cron)
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- 2. Création de la tâche de nettoyage
-- Supprime automatiquement toutes les extractions vectorielles vieilles de plus de 7 jours
-- Tourne tous les jours à minuit (0 0 * * *)
SELECT cron.schedule(
  'nettoyage-vecteurs-obsoletes',
  '0 0 * * *',
  $$ DELETE FROM clinical_trials_data_biobert WHERE created_at < NOW() - INTERVAL '7 days' $$
);
