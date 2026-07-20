# Clinical Protocols Standardization (AI-Powered) 🔬

Ce projet est une application complète (Data Engineering & Data Science) permettant l'ingestion, le traitement, l'indexation vectorielle, et l'extraction sémantique d'entités depuis des protocoles cliniques (fichiers PDF).

## 🏗️ Architecture du Projet (Pipeline Hybride V2)

L'application suit une architecture hautement optimisée, gérant intelligemment la collecte et l'extraction des données :

- **Collecte Intelligente (JSON vs PDF)** : Lorsqu'une pathologie est cherchée, l'app interroge l'API officielle ClinicalTrials V2. 
  - Si le texte structuré (`eligibilityCriteria`) est disponible, il est envoyé directement à l'IA pour une extraction foudroyante (< 1 seconde).
  - *Fallback (Plan B)* : Si l'utilisateur force l'extraction PDF, l'application scanne intelligemment les résultats de l'API v2 jusqu'à identifier des essais possédant réellement un fichier PDF attaché. Elle télécharge ensuite ces PDF originaux directement via le CDN public de ClinicalTrials.gov (ultra-rapide et robuste).
    > **💡 Point d'Ingénierie Fort (Recherche Intelligente & Scraping CDN) :** Remplacement stratégique du scraping par navigateur (Playwright) par une interrogation directe et intelligente des endpoints CDN via l'API v2. L'algorithme filtre à la volée les essais sans documents pour garantir un téléchargement réussi. Résultat : temps d'extraction divisés par 10, zéro dépendance complexe, et fiabilité totale sur des serveurs Cloud (Render).
- **Stockage Cloud (Supabase Storage)** : Dans le cas du "Plan B" (PDF téléchargé), le fichier est automatiquement sauvegardé dans un bucket public sur Supabase (`clinical_pdfs`) pour l'archivage.
- **Extraction RAG Hybride (BioBERT + Qwen)** : Le PDF ou le texte brut passe par notre pipeline IA. **BioBERT** fragmente le texte et trouve les paragraphes clés pertinents (Embedding vectoriel). Le LLM génératif **Qwen-7B** prend ensuite le relais pour lire ce contexte ultra-ciblé et formater la sortie en un fichier JSON structuré parfait.
- **Tableau de Bord Médical** : Les données structurées (Maladie, Médicaments, Critères d'inclusion) sont exposées au médecin via une interface Pandas/Streamlit (`st.dataframe`).
- **Base Vectorielle** : Supabase avec l'extension `pgvector` et un index HNSW.
- **Monitoring** : `MLflow` pour le suivi des performances (latence, prompts, JSON de sortie).

## 🚀 Démarrage Rapide

**1. Cloner le projet**
```bash
git clone https://github.com/Elkristobal59/clinicalapp.git
cd clinicalapp
```

**2. Lancer la Sainte Trinité des Terminaux (Lightning AI)**
Lors du démarrage de votre instance Lightning AI, vous devez lancer l'infrastructure backend (API + GPU) et l'outil de monitoring (MLflow).
Ouvrez 4 terminaux différents et lancez ces 4 commandes. **Il vous suffit de copier-coller, aucune URL n'est à modifier !**

**Terminal 1 : Le Cerveau (API & Modèles GPU)**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 : Le Pont API (Pour communiquer avec Streamlit)**
```bash
npx localtunnel --port 8000 --subdomain protocole-clinique-api
```

**Terminal 3 : L'Observatoire (Dashboard MLflow)**
```bash
mlflow ui --host 0.0.0.0 --port 5000 --disable-security-middleware
```

**Terminal 4 : Le Pont MLflow (Pour voir le Dashboard)**
```bash
npx localtunnel --port 5000 --subdomain mlflow-clinique-chris
```

✅ **C'est prêt !** 
- L'URL de l'API est fixée sur : `https://protocole-clinique-api.loca.lt` (à insérer dans Streamlit).
- Vos logs d'extractions en direct sont sur : `https://mlflow-clinique-chris.loca.lt` (cliquez sur "Click to Continue" pour y accéder).

## 💡 FAQ & Pièges classiques (Spécial Soutenance / Démo)

Si un membre du jury vous pose une question piège ou si vous avez un doute pendant la démo, voici les réponses :

> **❓ "Pourquoi MLflow affiche une date *Last modified* d'hier alors que je viens de faire une extraction ?"**
> L'interface de MLflow (page *Experiments*) affiche la date de **création du dossier**, pas la date de la dernière donnée insérée. C'est comme un classeur physique : on ne change pas l'étiquette quand on glisse une nouvelle feuille dedans.
> 👉 **Solution** : Cliquez sur le nom bleu `Clinical_Trials_Extraction` pour entrer dans le dossier. Vos extractions d'aujourd'hui sont bien là !

> **❓ "Je ne vois pas mes données dans MLflow, j'ai une erreur *Failed to load chart data* ou *Request Error* ?"**
> 1. Assurez-vous d'avoir cliqué sur le bouton **"Model training"** en haut à gauche de MLflow (le mode "GenAI" par défaut cherche des données complexes que nous n'utilisons pas).
> 2. Pour voir vos données (`disease`, `document`), cliquez sur le petit bouton **`+` (Show more columns)** à droite du tableau et cochez-les.

> **❓ "Si l'onglet 1 n'utilise que BioBERT, pourquoi l'API charge-t-elle le LLM Qwen au démarrage ?"**
> L'API doit être prête à servir les deux onglets simultanément. Si BioBERT est parfait pour l'extraction (Onglet 1), il est en revanche **incapable de formuler des phrases**. Le modèle Qwen est donc chargé en VRAM car il est le seul à pouvoir lire les résultats de BioBERT et générer une réponse en bon français pour le **Chatbot RAG (Onglet 2)**.

> **❓ "Le chatbot RAG commence à répéter les mêmes mots en boucle ?"**
> Les modèles de langage locaux (comme Qwen-7B) peuvent halluciner et boucler indéfiniment s'ils ne trouvent pas de réponse claire.
> 👉 **Solution** : Nous avons configuré une pénalité (`repetition_penalty = 1.1`) dans les paramètres d'inférence de l'API (`vLLM`) pour lui interdire formellement de répéter les mêmes séquences de mots. L'assistant est désormais stable.

> **❓ "Comment tester le téléchargement des PDF si tout passe rapidement par le JSON (Plan A) ?"**
> Pour les besoins de la démonstration, nous avons ajouté une option **"Ignorer le texte natif et forcer le scraping PDF (Plan B)"** directement dans l'interface Streamlit. Cochez-la pour forcer le lancement du navigateur fantôme et l'upload dans Supabase !

> **❓ "J'ai l'erreur `[Errno 98] Address already in use` quand je lance MLflow ou l'API ?"**
> Cela signifie que vous avez fermé le terminal avec la croix au lieu de faire `Ctrl+C`, le processus tourne donc toujours de manière invisible.
> 👉 **Solution** : Tapez `pkill -f "mlflow"` ou `pkill -f "uvicorn"` pour tuer les processus fantômes, puis relancez.

## 📂 Déploiement de l'Interface Web (Render / Docker)

L'interface client (Streamlit) est dockerisée pour être déployée sur Render, Heroku, etc.
Le projet utilise un fichier `requirements-frontend.txt` allégé pour le conteneur Docker afin d'éviter l'installation des librairies GPU lourdes sur le frontend.

```bash
docker-compose up --build
```
L'application sera accessible sur le port `8501`.

> 🛠️ **Dépannage Render (Déploiement Cloud)** :
> - **Redémarrages intempestifs (`Stopping...`)** : Fixez le port en ajoutant la variable d'environnement `PORT=8501` sur Render.
> - **Erreur `[Errno 24] inotify instance limit reached`** : Désactivez la surveillance en ajoutant la variable `STREAMLIT_SERVER_FILE_WATCHER_TYPE=none`.

### Infrastructure as Code (Terraform)
Le dossier `terraform/` contient les scripts pour générer la structure de la base de données Supabase automatiquement (`main.tf`, `schema.sql`).
```bash
cd terraform
terraform init
terraform apply
```
