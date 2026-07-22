# Défis Techniques et Résolutions (Demo Day)

Cette section documente les principales difficultés techniques rencontrées lors de la mise en place du pipeline d'entraînement et d'extraction de données, démontrant la capacité de l'équipe à surmonter les imprévus liés à l'utilisation d'outils et de modèles de pointe.

## 1. Instabilité des structures de données externes (HuggingFace Datasets)
**Le problème :** Lors de l'extraction des études cliniques, le schéma du dataset `bigbio/chia` a été mis à jour silencieusement sur HuggingFace.
- Les labels (types d'entités) qui étaient de simples chaînes de caractères (`"Condition"`) sont devenus des listes (`["Condition"]`).
- Le texte brut de l'étude (champ `text`) a disparu au profit d'une structure imbriquée `passages[0]["text"]`.
**La conséquence :** Le script d'extraction d'origine ne trouvait plus les données et générait un dataset vide (0 documents conservés).
**La résolution :** Rétro-ingénierie de la nouvelle structure JSON du dataset en direct et création de patchs pour rendre le parseur robuste aux différents formats (listes vs strings, champ `passages` vs `text`).

## 2. L'Enfer des dépendances et de l'environnement (HuggingFace TRL)
**Le problème :** Lors de la configuration du pipeline de Fine-Tuning (SFTTrainer), l'environnement Lightning a installé la toute dernière version de la librairie `trl` (v1.9.0 / v0.12+).
- Disparition soudaine de l'argument `dataset_text_field`.
- Disparition de l'argument `group_by_length`.
- Changement de nom strict pour le tokenizer (de `tokenizer` à `processing_class`).
**La conséquence :** Erreurs bloquantes en chaîne (`TypeError`) lors de l'initialisation de l'entraînement.
**La résolution :** Refactorisation complète du script d'entraînement pour utiliser les arguments fondamentaux (`TrainingArguments` standards) et s'adapter aux nouveaux standards stricts imposés par les développeurs de HuggingFace.

## 3. Conflits de précision mixte (PyTorch AMP & Qwen BFloat16)
**Le problème :** Le modèle de base `Qwen2.5-0.5B-Instruct` a été entraîné originellement en format `bfloat16`. Pour économiser la mémoire VRAM, nous avions paramétré l'entraînement en `float16` (`fp16=True`).
**La conséquence :** Lors de la première étape de rétropropagation (Gradient Descent), le `GradScaler` de PyTorch a détecté un conflit de types de tenseurs et a généré un crash brutal : `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`.
**La résolution :** Passage intégral du pipeline d'entraînement en `bfloat16` (`bf16=True`), ce qui permet d'utiliser le format natif du modèle Qwen et de désactiver le correcteur de gradient problématique de PyTorch.

## 4. Blocages asynchrones sur le Terminal (Git sur Windows)
**Le problème :** Lors de la synchronisation du code via Git, l'interface Git Credential Manager sous Windows a généré des invites de connexion invisibles en arrière-plan.
**La conséquence :** Les scripts de push sont restés figés indéfiniment, bloquant le transfert de code vers le cloud Lightning.
**La résolution :** Utilisation de scripts d'injection de code dynamique (patching via `python -c`) directement sur les serveurs distants pour by-passer Git de manière asynchrone sans ralentir l'avancée du projet.
