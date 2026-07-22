# 🚀 Tutoriel & Guide de Démarrage pour l'Équipe

Ce document rassemble toutes les commandes et procédures pour que chacun puisse tester, évaluer et faire tourner le projet sur sa machine ou sur Lightning AI.

---

## 🎨 1. Lancer l'Application Streamlit (Interface Visuelle)
L'application Streamlit regroupe toute la logique d'extraction (JSON/PDF) et le Chatbot RAG.
Elle inclut désormais le **Tableau Récapitulatif** (Summary Table) et les **Filtres Avancés** !

**Comment la lancer :**
1. Assurez-vous d'avoir fait un `git pull` pour récupérer les dernières modifications.
2. Ouvrez un terminal à la racine du projet (`stack_equipe`).
3. Lancez la commande suivante :
```bash
streamlit run app/streamlit_app.py
```
4. Une page web va s'ouvrir. Vous pouvez tester les différents filtres de recherche (Phase, InterventionName, etc.) et extraire les données.

*(Note par rapport au schéma d'architecture : le tableau récapitulatif avec l'analyse de complétude des JSON s'affiche directement en bas de page après l'extraction !)*

---

## 📊 2. Générer le Tableau de Complétude (Script de Jérémie)
Si vous voulez juste analyser la qualité des données remontées par l'API ClinicalTrials sans lancer l'interface graphique, un script dédié est disponible.

Il génère un fichier Excel/CSV avec les 10 champs obligatoires et vérifie si le PDF est présent ou si des données manquent (`N/A` ou `Absent`).

**Rechercher par mot-clé (ex: "Breast Cancer") :**
```bash
python scripts/generate_fields_summary.py --query "Breast Cancer"
```

**Rechercher par liste précise de NCT IDs :**
```bash
python scripts/generate_fields_summary.py --nct_ids "NCT01547806,NCT03047980,NCT04676152"
```
👉 Le fichier `summary_table_jeremie.csv` apparaîtra à la racine du projet.

---

## 🧠 3. Fine-Tuning du Modèle Qwen (Pour la personne sur Lightning AI)
L'entraînement du modèle d'Extraction d'Entités se fait exclusivement sur la machine GPU (Lightning AI).

**Étape 3.1 : Découpage du dataset (Séparation Train/Test)**
On isole les 129 études qui possèdent un PDF pour se les garder pour le Test, et on entraîne l'IA sur le reste.
```bash
python scripts/split_dataset.py
```

**Étape 3.2 : Connexion à HuggingFace (Requis pour Qwen)**
```bash
hf auth login
```
*(Collez le token secret en aveugle et faites Entrée).*

**Étape 3.3 : Lancement de l'Entraînement**
```bash
python scripts/finetune_qwen.py
```
*(L'entraînement dure environ 1 heure sur un GPU L4).*

---

## 📈 4. Évaluation du Modèle (Calcul des Scores)
Une fois l'entraînement terminé, on veut connaître la note (Précision, Rappel, Score F1) de l'IA sur les 129 études qu'elle n'a jamais vues.

**Faire passer le test à l'IA :**
```bash
python scripts/inference_qwen.py
```
*(L'IA va lire les textes du Test Set et extraire les entités).*

**Calculer la note finale :**
```bash
python scripts/evaluate_ner.py
```
👉 Les scores de **True Positives**, **False Positives** et **F1 Score** s'afficheront dans le terminal ! Vous pouvez les prendre en capture d'écran pour le Demo Day.
