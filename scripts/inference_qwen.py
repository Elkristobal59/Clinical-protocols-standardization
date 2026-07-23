"""
Script : inference_qwen.py (Phase 4 du Pipeline MLOps)
------------------------------------------------------
Rôle : Évaluer les performances du modèle fine-tuné sur le jeu de données de test (Test Set)
et calculer les métriques de précision (Score F1, Précision, Rappel).

🎓 Explication pour le jury (L'Inférence avec LoRA) :
Lorsqu'on a fine-tuné le modèle (étape 3), on a seulement sauvegardé un "Adaptateur LoRA" (très léger).
Pour utiliser le modèle, l'algorithme fait une opération de fusion (Merge) à la volée :
1. Il charge le "Cerveau de base" (Le modèle Qwen original, 14 Go).
2. Il charge la "Disquette de mise à jour" (Notre adaptateur LoRA CHIA, 40 Mo).
3. Il combine les deux en VRAM.
C'est hyper optimisé pour le Cloud : on peut avoir un seul gros cerveau de base, et charger plein de 
petits adaptateurs différents selon les besoins de l'hôpital !
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import re

# ---------------------------------------------------------
# ⚙️ CONFIGURATION
# ---------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"          # Le modèle de base
ADAPTER_DIR = "Elkristobal59/qwen-7b-chia-ner" # Notre adaptateur hébergé sur le Hub HuggingFace (ou en local)
TEST_DATA = "data/test_dataset.jsonl"          # Le jeu de données "Examen blanc" (jamais vu par le modèle)
OUTPUT_PREDS = "data/chia_predictions_qwen_7b.json"

def main():
    # ---------------------------------------------------------
    # 🧠 ÉTAPE 1 : CHARGEMENT DU MODÈLE (INFÉRENCE RAPIDE)
    # ---------------------------------------------------------
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 💡 Optimisation Inférence :
    # Contrairement à l'entraînement, on n'utilise PAS la compression 4-bit ici.
    # Pourquoi ? Car la carte graphique de production (L4 - 24Go VRAM) a assez de place
    # pour charger le modèle en bfloat16 (16-bits). Sans la compression 4-bit, 
    # le modèle génère le texte de manière BEAUCOUP plus rapide (10x à 20x plus rapide).
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # 🔌 Chargement de notre adaptateur médical
    print("Loading fine-tuned adapters...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval() # On bascule le modèle en mode "Évaluation" (désactive le calcul de gradients)

    # ---------------------------------------------------------
    # 📂 ÉTAPE 2 : PRÉPARATION DES DONNÉES DE TEST
    # ---------------------------------------------------------
    print("Loading test data...")
    test_examples = []
    with open(TEST_DATA, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_examples.append(json.loads(line))
                
    print(f"Running inference and evaluation on {len(test_examples)} examples...")
    
    # Compteurs pour la matrice de confusion (Évaluation Statistique)
    tp_total = 0 # Vrais Positifs (L'IA a trouvé la bonne entité)
    fp_total = 0 # Faux Positifs (L'IA a inventé une entité qui n'existe pas)
    fn_total = 0 # Faux Négatifs (L'IA a raté une entité qui était dans le texte)
    
    # ---------------------------------------------------------
    # 🚀 ÉTAPE 3 : GÉNÉRATION ET COMPARAISON (LA BOUCLE D'ÉVALUATION)
    # ---------------------------------------------------------
    for i, example in enumerate(tqdm(test_examples)):
        messages = example["messages"]
        
        # On extrait la question (System + User) et la réponse attendue (Assistant Ground Truth)
        prompt_messages = [msg for msg in messages if msg["role"] != "assistant"]
        expected_json_str = [msg for msg in messages if msg["role"] == "assistant"][0]["content"]
        
        try:
            expected_entities = json.loads(expected_json_str)
        except Exception:
            expected_entities = []
        
        # On prépare le prompt pour le modèle
        text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # 🎲 La Génération LLM (Inférence)
        with torch.no_grad(): # On coupe le calcul des gradients (économise énormément de RAM)
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048, # Le modèle a le droit de générer jusqu'à 2048 mots (tokens)
                do_sample=False,     # Température à 0 (Pas de créativité, on veut un comportement déterministe)
                pad_token_id=tokenizer.eos_token_id
            )
            
        # On ne garde que la réponse générée (on enlève la question du texte)
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 🛠️ POST-PROCESSING (Nettoyage de la réponse)
        # Parfois le LLM rajoute des balises Markdown (```json ... ```), il faut les nettoyer.
        try:
            if "```json" in response_text:
                json_part = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_part = response_text.split("```")[1].strip()
            else:
                json_part = response_text.strip()
                
            predicted_entities = json.loads(json_part)
        except Exception as e:
            # Si le modèle a halluciné un texte qui n'est pas du JSON valide, on loggue l'erreur
            if fp_total + fn_total < 10: 
                print(f"\n[DEBUG ERROR] JSON parse failed! Error: {e}")
                print(f"[DEBUG RAW OUTPUT] {repr(response_text[:500])}...\n")
            predicted_entities = []
            
        # ---------------------------------------------------------
        # 📊 ÉTAPE 4 : CALCUL DES SCORE SUR CE DOCUMENT
        # ---------------------------------------------------------
        # Format attendu : {"label": "...", "entity": "..."}
        # On utilise des 'Set' (Ensembles mathématiques Python) pour faire des comparaisons ultra-rapides.
        gold_set = set((ent.get("label", "").lower(), ent.get("entity", "").lower().strip()) for ent in expected_entities if isinstance(ent, dict))
        pred_set = set((ent.get("label", "").lower(), ent.get("entity", "").lower().strip()) for ent in predicted_entities if isinstance(ent, dict))
        
        # Intersection = L'IA a trouvé la bonne catégorie ET le bon mot
        tp = len(gold_set.intersection(pred_set))
        # Faux Positif = L'IA a trouvé un truc qui n'est pas dans les résultats attendus
        fp = len(pred_set - gold_set)
        # Faux Négatif = L'IA a raté quelque chose qui était dans la cible
        fn = len(gold_set - pred_set)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn

    # ---------------------------------------------------------
    # 🏆 ÉTAPE 5 : RÉSULTATS MLOps GLOBAUX
    # ---------------------------------------------------------
    # Précision = Sur 100 mots extraits par l'IA, combien étaient réellement corrects ?
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    # Rappel = Sur 100 mots cibles dans le texte, combien l'IA a-t-elle réussi à repérer ?
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    # Score F1 = La moyenne harmonique (l'équilibre parfait) entre Précision et Rappel
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*50)
    print("🎯 RÉSULTATS FINAUX SUR LE TEST SET (390 CHUNKS)")
    print("="*50)
    print(f"Vrais Positifs  (TP) : {tp_total}")
    print(f"Faux Positifs   (FP) : {fp_total}")
    print(f"Faux Négatifs   (FN) : {fn_total}")
    print("-" * 50)
    print(f"Précision (Precision) : {precision:.4f} ({(precision*100):.1f}%)")
    print(f"Rappel    (Recall)    : {recall:.4f} ({(recall*100):.1f}%)")
    print(f"Score F1  (F1 Score)  : {f1:.4f} ({(f1*100):.1f}%)")
    print("="*50)

if __name__ == "__main__":
    main()
