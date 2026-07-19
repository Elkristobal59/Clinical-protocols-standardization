# Clinical Protocols Standardization (AI-Powered) 🔬

Ce projet est une application complète (Data Engineering & Data Science) permettant l'ingestion, le traitement, l'indexation vectorielle, et l'extraction sémantique (LLM) de protocoles cliniques (fichiers PDF) depuis ClinicalTrials.gov.

## 🏗️ Architecture du Projet

L'application suit une architecture Client/Serveur découplée :
- **Scraping** : Playwright (local) pour la récupération automatisée des PDF de protocoles cliniques.
- **Base Vectorielle** : Supabase avec l'extension `pgvector` et un index HNSW pour la recherche de similarité ultra-rapide.
- **Inférence & RAG (Serveur)** : API FastAPI hébergée sur **Lightning AI** (GPU). Encode les données avec `BioBERT` et génère les réponses via le LLM `Qwen-1.5B`.
- **Monitoring** : `MLflow` pour tracer les performances, les requêtes (temps, modèle, prompt, json de sortie).
- **Interface Client** : `Streamlit` dockerisé pour interagir avec le serveur.

## 🚀 Fonctionnalités

1. **Onglet 1 : Ingestion & Extraction**
   - Entrez une maladie (ex: *Cardiology*, *Breast Cancer*).
   - L'application scrape automatiquement ClinicalTrials, télécharge les PDF, les envoie au serveur GPU, génère les embeddings et extrait les informations cibles (Condition, Médicaments, Critères d'inclusion) dans un format JSON structuré.
   - **Mise à jour NER (Named Entity Recognition)** : L'extraction s'appuie désormais sur un "prompt strict" issu du benchmark Chia, permettant de classer finement les entités (Condition, Drug, Procedure, Measurement, etc.) et de réduire drastiquement les hallucinations.
2. **Onglet 2 : Chatbot RAG (Retrieval-Augmented Generation)**
   - Posez des questions en langage naturel sur la base de connaissances constituée des protocoles cliniques.
   - Les réponses générées sont basées **uniquement** sur les extraits sémantiques trouvés (affichage des sources de décision).

## 🛠️ Stack Technique

- **Backend / Serveur** : FastAPI, PyTorch, Transformers, Uvicorn.
- **Modèles IA** : `dmis-lab/biobert-v1.1` (Embeddings), `Qwen/Qwen1.5-1.8B-Chat` (LLM).
- **Frontend / Client** : Streamlit, Playwright.
- **Data & Ops** : PostgreSQL (Supabase), MLflow, Docker, Terraform.

## 📂 Déploiement

### Déploiement du Client (Docker / Render)
L'interface utilisateur est entièrement dockerisée et prête à être déployée (Render, Heroku, DigitalOcean...) :
Le projet dispose d'un `requirements-frontend.txt` allégé pour le conteneur Docker afin d'éviter l'installation inutile des dépendances GPU lourdes (PyTorch, vLLM) sur le serveur web.
```bash
docker-compose up --build
```
L'application sera accessible sur le port `8501`.

> 🛠️ **Dépannage Render (Déploiement Cloud)** :
> - **Redémarrages intempestifs (`Stopping...`)** : Render détecte parfois les ports ouverts par Playwright (Chrome) et redémarre le conteneur par erreur. Fixez le port en ajoutant la variable d'environnement `PORT=8501` sur Render.
> - **Erreur `[Errno 24] inotify instance limit reached`** : Le système manque de mémoire/processus pour surveiller les fichiers (hot-reloading Streamlit). Désactivez la surveillance en ajoutant la variable `STREAMLIT_SERVER_FILE_WATCHER_TYPE=none` (déjà patché dans le Dockerfile).

### Déploiement Serveur (Lightning AI)
1. Installez les dépendances (`requirements.txt`).
2. Démarrez l'API :
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   > 💡 **Smart Fallback (vLLM)** : L'API détecte automatiquement votre matériel au lancement. 
   > - **Si un GPU est détecté** : Le serveur active le moteur haute-performance **vLLM** (paramétré pour prendre 70% de la VRAM, laissant 30% pour BioBERT), garantissant des vitesses de génération fulgurantes.
   > - **Si vous êtes sur CPU** : L'API bascule silencieusement sur la librairie `transformers` native pour permettre le développement et les tests locaux sans planter (en contournant les bugs liés à la librairie `accelerate`).

   > 🛠️ **Dépannage (Troubleshooting)** :
   > - **Erreur `numpy.dtype size changed`** (liée à **MLflow** ou **Pandas**) : C'est un conflit de versions dû à l'installation de vLLM. Résolvez-le en forçant la version 1.x de Numpy :
   >   ```bash
   >   pip install "numpy<2"
   >   ```
   > - **Erreur `psycopg2.OperationalError: connection to server on socket...`** : L'API ne trouve pas votre URL Supabase car le fichier `.env` n'existe pas sur la machine distante. Créez-le à la racine avec vos identifiants :
   >   ```bash
   >   echo 'SUPABASE_DATABASE_URL="postgresql://postgres.[VOTRE_ID]:[MOT_DE_PASSE]@aws-0-eu-west-1.pooler.supabase.com:6543/postgres"' > .env
   >   ```
   > - **Erreur `AssertionError: Found 2 libcudnn.so.x`** : Lors du passage sur GPU, PyTorch peut télécharger des pilotes NVIDIA en double qui font planter vLLM. Nettoyez avec :
   >   ```bash
   >   pip uninstall -y nvidia-cudnn-cu11 nvidia-cudnn-cu12
   >   pip install nvidia-cudnn-cu12
   >   ```
   > - **Erreur `address already in use`** (sur le port 8000) : Un ancien serveur uvicorn tourne en tâche de fond. Tuez-le avec :
   >   ```bash
   >   fuser -k 8000/tcp
   >   ```

3. Exposez le port via localtunnel :
   - **Option A (Recommandée)** : Utilisez un sous-domaine fixe pour ne pas avoir à modifier la configuration de votre frontend tous les jours. Choisissez un nom **unique** :
     ```bash
     lt --port 8000 --subdomain api-clinique-votre-nom
     ```
     *(💡 Note : Si le sous-domaine est déjà pris ou bloqué par un ancien crash, localtunnel l'ignorera et vous donnera une adresse aléatoire à la place).*
   - **Option B (Aléatoire)** : Génère une URL unique à chaque fois :
     ```bash
     lt --port 8000
     ```

4. Renseignez l'URL générée (ex: `https://api-clinique-votre-nom.loca.lt`) dans la variable d'environnement `LIGHTNING_AI_API_URL` de votre client web (sur Render), ou modifiez-la directement dans le champ de texte sur l'interface Streamlit.

### Infrastructure as Code (Terraform)
Le dossier `terraform/` contient les scripts pour générer la structure de la base de données Supabase automatiquement (`main.tf`, `schema.sql`).
```bash
cd terraform
terraform init
terraform apply
```
