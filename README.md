# Clinical Protocols Standardization (AI-Powered) 🔬

Ce projet est une application complète (Data Engineering & Data Science) permettant l'ingestion, le traitement, l'indexation vectorielle, et l'extraction sémantique d'entités depuis des protocoles cliniques (fichiers PDF).

## 🏗️ Architecture du Projet (Pipeline Hybride)

L'application suit une architecture hautement optimisée, séparant les tâches de compréhension et de génération :

- **Extraction d'Entités (Onglet 1 - Full BioBERT)** : La standardisation des données (NER) est désormais **100% basée sur BioBERT** (`biobert-chia-ner`). Contrairement à une extraction classique par LLM, ce modèle de classification de tokens extrait les entités (Condition, Drug, Measurement) quasi instantanément et sans hallucination. Le modèle Qwen a été *totalement retiré* de cette étape pour des raisons de performance.
- **Assistant Conversationnel RAG (Onglet 2 - BioBERT + Qwen)** : Le Chatbot RAG utilise une architecture hybride. **BioBERT** agit comme encodeur pour créer les vecteurs (embeddings) et trouver les informations dans la base. Le LLM léger **Qwen-1.5B** prend ensuite le relais *uniquement* pour générer la réponse en langage naturel à partir de ces extraits.
- **Base Vectorielle** : Supabase avec l'extension `pgvector` et un index HNSW.
- **Serveur d'Inférence** : API FastAPI hébergée sur **Lightning AI** (GPU).
- **Monitoring** : `MLflow` pour le suivi des performances (latence, prompts, JSON de sortie).

## 🚀 Utilisation de vLLM (Accélération GPU)

Pour l'onglet RAG, la génération de texte par le LLM (Qwen-1.5B) est propulsée par **vLLM**, un moteur d'inférence ultra-rapide.

- **Smart Fallback** : L'API détecte automatiquement votre matériel au lancement. Si un GPU est présent, vLLM est activé. Sinon, l'API bascule sur la librairie `transformers` native (plus lente mais fonctionnelle sur CPU).
- **Gestion de la VRAM** : vLLM est paramétré pour utiliser `70%` de la mémoire vidéo (`gpu_memory_utilization=0.7`). Cela garantit qu'il reste toujours `30%` de VRAM disponible pour faire tourner BioBERT en parallèle sans crash (OOM).
- **Bénéfice** : vLLM utilise la technique du *PagedAttention* pour gérer le cache KV de manière optimale, rendant les réponses du Chatbot fluides et instantanées.

## 📊 Suivi des Expériences (MLflow)

Chaque action dans l'application (extraction d'un PDF ou question posée au RAG) est loguée dans **MLflow**.

**Pour ouvrir l'interface MLflow :**
1. Sur votre instance Lightning AI, ouvrez un nouveau terminal.
2. Tapez la commande suivante pour lancer le dashboard :
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   ```
3. Si vous avez besoin d'exposer ce port à internet pour y accéder depuis votre navigateur (comme pour l'API), utilisez un autre tunnel local :
   ```bash
   npx localtunnel --port 5000 --subdomain mlflow-clinique
   ```
4. **Dans l'interface MLflow**, vous pourrez voir en temps réel :
   - Le temps de latence de chaque requête API.
   - Les documents exacts traités et les paramètres (maladie).
   - Les prompts complets envoyés au RAG et les réponses.
   - Les JSON finaux générés par l'extraction BioBERT.

## 📂 Déploiement

### Déploiement du Client (Docker / Render)
L'interface utilisateur est entièrement dockerisée et prête à être déployée (Render, Heroku, DigitalOcean...) :
Le projet dispose d'un `requirements-frontend.txt` allégé pour le conteneur Docker afin d'éviter l'installation inutile des dépendances GPU lourdes sur le serveur web.
```bash
docker-compose up --build
```
L'application sera accessible sur le port `8501`.

> 🛠️ **Dépannage Render (Déploiement Cloud)** :
> - **Redémarrages intempestifs (`Stopping...`)** : Fixez le port en ajoutant la variable d'environnement `PORT=8501` sur Render.
> - **Erreur `[Errno 24] inotify instance limit reached`** : Désactivez la surveillance en ajoutant la variable `STREAMLIT_SERVER_FILE_WATCHER_TYPE=none`.

### Déploiement Serveur (Lightning AI)
1. Installez les dépendances (`requirements.txt`).
2. Démarrez l'API (dans un premier terminal) :
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
3. Exposez le port via localtunnel (dans un second terminal) :
   ```bash
   npx localtunnel --port 8000 --subdomain api-clinique-votre-nom
   ```
   *(💡 Note : `lt` est l'abréviation de `localtunnel`. `npx localtunnel` télécharge et exécute le tunnel à la volée s'il n'est pas installé globalement).*

4. Renseignez l'URL générée (ex: `https://api-clinique-votre-nom.loca.lt`) dans la variable d'environnement `LIGHTNING_AI_API_URL` de votre client web (sur Render).

### Infrastructure as Code (Terraform)
Le dossier `terraform/` contient les scripts pour générer la structure de la base de données Supabase automatiquement (`main.tf`, `schema.sql`).
```bash
cd terraform
terraform init
terraform apply
```
