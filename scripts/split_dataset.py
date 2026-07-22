import json
import random
import os

# Reproductibilité (utile si on veut re-mélanger plus tard, mais ici on va isoler les PDF)
random.seed(42)

def convert_to_chatml(doc_dict):
    """
    Convertit un dictionnaire document (text + entities) au format conversationnel (ChatML).
    """
    text = doc_dict["text"]
    entities = doc_dict["entities"]
    
    # Prompt utilisateur (instruction + texte source)
    user_prompt = f"Extract all relevant clinical entities from the following text and format them as JSON. The allowed entity types are: Condition, Drug, Procedure, Measurement, Value, Temporal, Observation, Person, Device.\n\nText: {text}"
    
    # Formatage de la réponse attendue en JSON propre
    expected_output = []
    for ent in entities:
        expected_output.append({
            "entity": ent["text"],
            "label": ent["label"]
        })
        
    assistant_response = json.dumps(expected_output, ensure_ascii=False)
    
    # Structure ChatML (OpenAI/Qwen)
    chat_format = {
        "messages": [
            {"role": "system", "content": "You are a medical AI assistant specialized in clinical trial named entity recognition (NER). You extract key entities precisely and format them in JSON."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }
    
    return chat_format

def main():
    input_path = os.path.join("data", "chia_gold_standard_v2.json")
    pdf_dir = os.path.join("data", "chia_pdfs")
    
    if not os.path.exists(input_path):
        print(f"❌ Erreur : {input_path} est introuvable. Lancez extract_full_chia.py d'abord.")
        return

    # 1. Identifier les études qui ont un PDF associé
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

    # 2. Charger toutes les données extraites
    with open(input_path, "r", encoding="utf-8") as f:
        dataset_v2 = json.load(f)
        
    print(f"Chargement de {len(dataset_v2)} blocs de texte (chunks).")
    
    # Rassembler par NCT ID
    studies = {} 
    for doc in dataset_v2:
        nct_id = doc["file"].split("_")[0]
        if nct_id not in studies:
            studies[nct_id] = []
        studies[nct_id].append(doc)
        
    unique_nct = list(studies.keys())
    print(f"Nombre total d'études uniques (NCT IDs) valides : {len(unique_nct)}")
    
    # 3. Séparation Stratégique selon la règle de Jérémie
    train_ncts = []
    test_ncts = []
    
    for nct in unique_nct:
        if nct in pdf_nct_ids:
            # Si l'étude a un PDF, elle part dans le Test (pour évaluation globale plus tard)
            test_ncts.append(nct)
        else:
            # Sinon, elle sert de matériel d'entraînement
            train_ncts.append(nct)
            
    print(f"\n📊 NOUVELLE RÉPARTITION (Stratégie PDF) :")
    print(f" -> {len(train_ncts)} études pour le Train (sans PDF)")
    print(f" -> {len(test_ncts)} études pour le Test (avec PDF)")
    
    # 4. Conversion et Sauvegarde
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
