"""
Script : split_dataset.py (Phase 2 du Pipeline MLOps)
-----------------------------------------------------
Rôle : Séparer la donnée brute en deux sets distincts (Train et Test) en évitant le Data Leakage, 
et formater les données au format "ChatML" attendu par le LLM Qwen.

🎓 Explication pour le jury :
1. Pourquoi séparer sur les "NCT_ID" et non pas aléatoirement ? 
   Si on mélange tout, un même essai clinique (NCT_ID) pourrait se retrouver à la fois dans l'entraînement 
   et dans le test. Le modèle apprendrait "par cœur" l'essai au lieu de généraliser. On appelle ça le Data Leakage.
2. Pourquoi le format ChatML ?
   Les LLM récents (Qwen, Llama 3) sont entraînés sous forme de dialogue (Système, User, Assistant). 
   On doit transformer nos JSON CHIA bruts en faux dialogues pour que le modèle apprenne à "répondre" correctement.
"""

import json
import random
import os

# Reproductibilité : Fixer le seed garantit que si on relance le script, on aura toujours la même répartition.
random.seed(42)

def convert_to_chatml(doc_dict):
    """
    Convertit un dictionnaire document (text + entities) au format conversationnel (ChatML).
    """
    text = doc_dict["text"]
    entities = doc_dict["entities"]
    
    # 🗣️ PROMPT INSTRUCTIONNEL (User)
    # C'est la consigne qu'on donnera au modèle en production. Il apprend à l'associer au format de sortie.
    user_prompt = f"Extract all relevant clinical entities from the following text and format them as JSON. The allowed entity types are: Condition, Drug, Procedure, Measurement, Value, Temporal, Observation, Person, Device.\n\nText: {text}"
    
    # 🤖 RÉPONSE ATTENDUE (Assistant)
    # C'est la "Ground Truth" (Vérité terrain). On formatte le JSON exact que l'on veut que le modèle génère.
    expected_output = []
    for ent in entities:
        expected_output.append({
            "entity": ent["text"],
            "label": ent["label"]
        })
        
    assistant_response = json.dumps(expected_output, ensure_ascii=False)
    
    # 📦 ASSEMBLAGE CHATML
    # Format standard attendu par les librairies de Fine-Tuning modernes (comme TRL ou Unsloth).
    chat_format = {
        "messages": [
            {"role": "system", "content": "You are a medical AI assistant specialized in clinical trial named entity recognition (NER). You extract key entities precisely and format them in JSON."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response} # Ce que le modèle doit apprendre à prédire
        ]
    }
    
    return chat_format

def main():
    input_path = os.path.join("data", "chia_gold_standard_v2.json")
    pdf_dir = os.path.join("data", "chia_pdfs")
    
    if not os.path.exists(input_path):
        print(f"❌ Erreur : {input_path} est introuvable. Lancez extract_full_chia.py d'abord.")
        return

    # 1. 🔍 IDENTIFICATION DES PDFS
    # Règle métier : On veut pouvoir faire des démos sur des PDF complets.
    # On va donc s'assurer que les études dont on a le PDF sont réservées pour le TEST (l'examen), 
    # et non pour l'entraînement (la révision).
    pdf_nct_ids = set()
    if os.path.exists(pdf_dir):
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                nct_id = filename.split("_")[0]
                pdf_nct_ids.add(nct_id)
        print(f"✅ Trouvé {len(pdf_nct_ids)} études avec des documents PDF complets.")
    else:
        print(f"⚠️ Attention : Le dossier {pdf_dir} n'existe pas. Assurez-vous d'avoir téléchargé les PDFs.")
        return

    # 2. 📂 CHARGEMENT DES DONNÉES
    with open(input_path, "r", encoding="utf-8") as f:
        dataset_v2 = json.load(f)
        
    print(f"Chargement de {len(dataset_v2)} blocs de texte (chunks).")
    
    # 🔗 REGROUPEMENT PAR ÉTUDE (NCT_ID)
    # Très important pour éviter le Data Leakage (fuite de données) entre Train et Test.
    studies = {} 
    for doc in dataset_v2:
        nct_id = doc["file"].split("_")[0]
        if nct_id not in studies:
            studies[nct_id] = []
        studies[nct_id].append(doc)
        
    unique_nct = list(studies.keys())
    print(f"Nombre total d'études uniques (NCT IDs) valides : {len(unique_nct)}")
    
    # 3. ✂️ SÉPARATION STRATÉGIQUE (Train / Test Split)
    train_ncts = []
    test_ncts = []
    
    for nct in unique_nct:
        if nct in pdf_nct_ids:
            # Si l'étude a un PDF, elle part dans le Test (On la garde pour évaluer l'API/Streamlit plus tard)
            test_ncts.append(nct)
        else:
            # Sinon, elle sert de matériel d'entraînement (Train)
            train_ncts.append(nct)
            
    print(f"\n📊 NOUVELLE RÉPARTITION (Stratégie PDF) :")
    print(f" -> {len(train_ncts)} études pour le Train (sans PDF)")
    print(f" -> {len(test_ncts)} études pour le Test (avec PDF)")
    
    # 4. 🔄 CONVERSION EN CHATML ET SAUVEGARDE (JSONL)
    # On sauvegarde en JSONL (un JSON par ligne). C'est le standard de l'industrie pour les très gros datasets LLM
    # car cela permet de les lire ligne par ligne (streaming) sans surcharger la mémoire RAM.
    train_chatml = []
    for nct in train_ncts:
        for chunk in studies[nct]:
            train_chatml.append(convert_to_chatml(chunk))
            
    test_chatml = []
    for nct in test_ncts:
        for chunk in studies[nct]:
            test_chatml.append(convert_to_chatml(chunk))
            
    train_path = os.path.join("data", "train_dataset.jsonl")
    test_path = os.path.join("data", "test_dataset.jsonl")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_chatml:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_chatml:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("\n✅ Fichiers JSONL régénérés avec succès :")
    print(f" - {train_path} ({len(train_chatml)} exemples chunks)")
    print(f" - {test_path} ({len(test_chatml)} exemples chunks)")

if __name__ == "__main__":
    main()
