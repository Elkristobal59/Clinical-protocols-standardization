import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import re

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "models/qwen_chia_finetuned"
TEST_DATA = "data/test_dataset.jsonl"
OUTPUT_PREDS = "data/chia_predictions_qwen.json"

def main():
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Suppression de la quantification 4-bit pour l'inférence. 
    # Le modèle 0.5B fait ~1Go en bf16, il tiendra largement sur la L4 (24Go)
    # et l'inférence sera 10x à 20x plus rapide !
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Charger les poids finetunés (LoRA)
    print("Loading fine-tuned adapters...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    print("Loading test data...")
    test_examples = []
    with open(TEST_DATA, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_examples.append(json.loads(line))
                
    print(f"Running inference and evaluation on {len(test_examples)} examples...")
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    for i, example in enumerate(tqdm(test_examples)):
        messages = example["messages"]
        
        # Extraire le prompt (sans la réponse attendue)
        prompt_messages = [msg for msg in messages if msg["role"] != "assistant"]
        expected_json_str = [msg for msg in messages if msg["role"] == "assistant"][0]["content"]
        
        try:
            expected_entities = json.loads(expected_json_str)
        except Exception:
            expected_entities = []
        
        # Préparer l'entrée pour Qwen
        text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Générer
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parser la réponse JSON du modèle
        try:
            if "```json" in response_text:
                json_part = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_part = response_text.split("```")[1].strip()
            else:
                json_part = response_text.strip()
                
            predicted_entities = json.loads(json_part)
        except Exception:
            predicted_entities = []
            
        # Format attendu : {"label": "...", "entity": "..."}
        gold_set = set((ent.get("label", "").lower(), ent.get("entity", "").lower().strip()) for ent in expected_entities if isinstance(ent, dict))
        pred_set = set((ent.get("label", "").lower(), ent.get("entity", "").lower().strip()) for ent in predicted_entities if isinstance(ent, dict))
        
        tp = len(gold_set.intersection(pred_set))
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn

    # Calcul des métriques finales
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
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
