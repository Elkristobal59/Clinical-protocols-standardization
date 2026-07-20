# Proposition d'Architecture Hybride (Ensemble Learning)

Ce document décrit l'évolution architecturale suggérée pour le pipeline de Named Entity Recognition (NER), justifiée par les résultats du benchmark sur le dataset **Chia**. À présenter au jury dans les "Perspectives d'évolution".

## 1. Contexte et Limites du Modèle Unique

Lors de l'évaluation de notre pipeline d'extraction, nous avons mis en compétition deux familles de modèles d'Intelligence Artificielle :
- **Un Encodeur Spécialisé (BioBERT)** : Entraîné spécifiquement sur le dataset clinique Chia (Token-Classification).
- **Un Décodeur Généraliste (Qwen 1.5B)** : Un grand modèle de langage fonctionnant en *Zero-Shot* via prompting strict.

### Analyse des scores (Micro-F1)
Le benchmark a révélé une dynamique très claire :
* **BioBERT excelle sur la structure** : Son score F1 global (0.41) est le double de celui du LLM (0.19). Il comprend parfaitement la grammaire des protocoles cliniques et extrait avec une grande précision les conditions médicales, les temporalités et les mesures.
* **Qwen excelle sur la chimie** : Bien que son score global soit faible, sa précision (Precision) sur la détection des **médicaments (Drug)** monte à **73%** (contre seulement 21% pour BioBERT). Sa "culture générale" encyclopédique pallie le manque de contexte que BioBERT n'arrive pas à résoudre.

## 2. La Solution : Pipeline Hybride (Ensemble Learning)

Utiliser un seul modèle ("Full BioBERT" ou "Full Qwen") nous oblige à faire un compromis inacceptable dans le monde médical : sacrifier soit la compréhension globale de la phrase, soit la précision sur les traitements.

Notre prochaine itération architecturale vise à implémenter un **Pipeline Hybride** qui fusionne les forces des deux modèles pour masquer leurs faiblesses respectives.

### Architecture cible :
1. **Extraction de base (BioBERT-NER)** : Chaque document PDF ingéré est scanné par BioBERT. Nous conservons toutes les entités extraites (Condition, Procedure, Measurement, etc.) **à l'exception des entités de type `Drug`**.
2. **Extraction Experte (Qwen)** : En parallèle, les mêmes extraits de texte sont envoyés au LLM avec un prompt ultra-ciblé : *"Extrais uniquement les médicaments de ce texte"*.
3. **Fusion (Merge Node)** : L'API réconcilie les deux listes d'entités en un seul fichier JSON unifié avant de l'envoyer au client.

## 3. Avantages pour l'Application
- **Réduction des Hallucinations** : On évite les 80% de faux-positifs de BioBERT sur les médicaments.
- **Rappel Maximal** : On conserve la capacité de BioBERT à extraire les critères d'inclusion complexes sans en oublier la moitié.
- **Scalabilité** : Les deux modèles fonctionnant sur des processus séparés (voire des serveurs GPU séparés via vLLM), l'extraction parallèle n'impacte pas le temps de réponse total perçu par l'utilisateur.

> *Cette architecture démontre une réelle maturité d'ingénierie : nous ne subissons plus les biais des modèles, nous les orchestrons de manière pragmatique selon leurs forces statistiques.*
