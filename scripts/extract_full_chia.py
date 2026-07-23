"""
Script : extract_full_chia.py (Phase 1 du Pipeline MLOps)
---------------------------------------------------------
Rôle : Collecter la donnée brute (Essais cliniques annotés) depuis HuggingFace et la formater 
pour l'entraînement. 

🎓 C'est la première étape du pipeline Data Engineering : on ingère la donnée source, 
on la nettoie (sélection des entités utiles) et on la sécurise (Holdout).
"""

import json
import re
import os
from datasets import load_dataset

# ---------------------------------------------------------
# 🛡️ 1. SÉCURITÉ DATA LEAKAGE (Holdout Set)
# ---------------------------------------------------------
# Ces 5 études ont été annotées manuellement (par Arnaud) pour l'évaluation finale.
# Il est CRITIQUE que le modèle ne les voit JAMAIS pendant l'entraînement. 
# C'est ce qu'on appelle prévenir le "Data Leakage" (Fuite de données).
HOLDOUT_NCT = {"NCT02145403", "NCT02541383", "NCT03346538", "NCT04676152", "NCT04915729"}

# ---------------------------------------------------------
# 🎯 2. SÉLECTION DES ENTITÉS MÉTIERS
# ---------------------------------------------------------
# Le dataset CHIA d'origine contient des dizaines de types d'annotations. 
# Pour notre application clinique, on restreint le périmètre à 9 entités essentielles.
TARGET_ENTITIES = {
    "Condition", "Drug", "Procedure", "Measurement", 
    "Value", "Temporal", "Observation", "Person", "Device"
}

def main():
    # 📥 INGESTION : Téléchargement du dataset depuis HuggingFace (bigbio/chia)
    print("Chargement du dataset CHIA depuis HuggingFace (bigbio/chia)...")
    try:
        # trust_remote_code=True est nécessaire car le dataset utilise un script Python personnalisé
        ds = load_dataset("bigbio/chia", "chia_bigbio_kb", split="train", trust_remote_code=True)
    except TypeError:
        # Fallback pour les anciennes versions de la librairie datasets
        ds = load_dataset("bigbio/chia", "chia_bigbio_kb", split="train")

    dataset_v2 = []
    
    print("Filtrage des études et extraction des entités...")
    # 🔄 TRANSFORMATION (La boucle d'ETL)
    for ex in ds:
        # 1. Extraction de l'identifiant du document
        doc_id = ex.get("document_id") or ex.get("id") or ""
        
        # 2. Identification du format standard NCT (ex: NCT01410890)
        m = re.search(r"NCT\d+", doc_id)
        if not m:
            continue
            
        nct_id = m.group(0)
        
        # 🛑 APPLICATION DU FILTRE ANTI-LEAKAGE
        # Si le NCT fait partie des études secrètes d'Arnaud, on l'ignore purement et simplement.
        if nct_id in HOLDOUT_NCT:
            print(f"⚠️  Étude {nct_id} ignorée (Réservée pour le Holdout Set d'Arnaud)")
            continue

        # 3. Récupération du texte brut
        # Le format BigBio place souvent le texte dans un sous-dictionnaire "passages"
        passages = ex.get("passages", [])
        text_list = passages[0].get("text") if passages else ex.get("text")
        
        if not text_list:
            continue
            
        # Consolidation du texte s'il est découpé en liste
        source_text = " ".join(text_list) if isinstance(text_list, list) else text_list

        entities = []
        # 4. Parcours de toutes les entités annotées (les "labels") dans le document
        for e in ex.get("entities", []):
            label_raw = e.get("type")
            label = label_raw[0] if isinstance(label_raw, list) and len(label_raw) > 0 else label_raw
            
            # On ignore les catégories qui ne nous intéressent pas
            if label not in TARGET_ENTITIES:
                continue
                
            # Récupération des offsets (les coordonnées du mot dans le texte : début et fin)
            offsets = e.get("offsets", [])
            if not offsets:
                continue
            
            start_offset = offsets[0][0]
            end_offset = offsets[0][1]
            
            # Le mot exact qui a été annoté
            ent_text_list = e.get("text", [])
            ent_text = " ".join(ent_text_list) if isinstance(ent_text_list, list) else ent_text_list
            
            # Ajout à notre structure de données épurée
            entities.append({
                "id": e.get("id"),
                "label": label,
                "text": ent_text,
                "start_offset": start_offset,
                "end_offset": end_offset
            })
            
        # 💾 SAUVEGARDE EN MÉMOIRE
        # Si le document contenait au moins une entité intéressante, on le garde pour l'entraînement
        if entities:
            dataset_v2.append({
                "file": doc_id,
                "text": source_text,
                "entities": entities
            })

    # 📤 LOAD (Sauvegarde du résultat final)
    output_path = os.path.join("data", "chia_gold_standard_v2.json")
    print(f"\nExtraction terminée. {len(dataset_v2)} sous-documents conservés (hors Holdout).")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # On exporte le tout dans un fichier JSON "Gold Standard" propre et sécurisé
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_v2, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Fichier sauvegardé : {output_path}")

if __name__ == "__main__":
    main()
