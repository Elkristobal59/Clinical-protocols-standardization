"""
Script : finetune_qwen.py (Phase 3 du Pipeline MLOps)
-----------------------------------------------------
Rôle : Entraîner (Fine-Tuner) le modèle de langage Qwen-7B sur nos données CHIA.

🎓 Explication pour le jury (La magie QLoRA) :
Un modèle de 7 Milliards de paramètres (7B) pèse environ 14 Go en mémoire (VRAM). 
L'entraîner classiquement (Full Fine-Tuning) demanderait plus de 80 Go de VRAM (plusieurs milliers d'euros).
Notre solution FinOps (QLoRA) :
1. "Q" (Quantization - 4-bit) : On compresse les poids du modèle de 16-bits à 4-bits. Le modèle passe de 14 Go à ~5 Go !
2. "LoRA" (Low-Rank Adaptation) : On "gèle" le cerveau du modèle, et on lui greffe de toutes petites "couches" supplémentaires (Adaptateurs). On n'entraîne que ces adaptateurs (qui pèsent 40 Mo au lieu de 14 Go).
Résultat : On peut entraîner une IA de pointe sur une simple carte graphique abordable (ex: L4) en quelques heures.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------
# ⚙️ CONFIGURATION DU MODÈLE
# ---------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" # Modèle de base open-source
DATASET_PATH = "data/train_dataset.jsonl"
OUTPUT_DIR = "models/qwen_7b_chia_finetuned"

def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # ---------------------------------------------------------
    # 📉 ÉTAPE 1 : LA QUANTIZATION (4-bit)
    # ---------------------------------------------------------
    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                 # Active la compression 4-bit
        bnb_4bit_use_double_quant=True,    # Double compression pour gagner encore plus de mémoire
        bnb_4bit_quant_type="nf4",         # Format NormalFloat4 (optimisé pour les poids des LLMs)
        bnb_4bit_compute_dtype=torch.float16 # Les calculs (maths) restent en 16-bits pour la précision
    )
    
    print(f"Loading model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",                 # Place automatiquement le modèle sur le GPU disponible
        dtype=torch.float16
    )
    model.config.use_cache = False         # Obligatoire de désactiver le cache pendant l'entraînement
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token # On définit le token de remplissage (padding)
    
    # ---------------------------------------------------------
    # 🧠 ÉTAPE 2 : CONFIGURATION LoRA (Les Adaptateurs)
    # ---------------------------------------------------------
    peft_config = LoraConfig(
        r=16,                              # La taille de la matrice de l'adaptateur (plus c'est gros, plus ça apprend, plus ça consomme)
        lora_alpha=32,                     # Le "poids" accordé à l'adaptateur par rapport au modèle d'origine
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # On cible les couches d'Attention du Transformer
        lora_dropout=0.05,                 # Évite l'overfitting (désactive 5% des neurones aléatoirement)
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # ---------------------------------------------------------
    # 📝 ÉTAPE 3 : FORMATAGE DU PROMPT
    # ---------------------------------------------------------
    def formatting_prompts_func(example):
        messages = example["messages"]
        # Transforme le format "ChatML" (JSON) en une longue chaîne de caractères 
        # que le modèle peut lire et comprendre avec ses balises spéciales (<|im_start|>)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}
        
    dataset = dataset.map(formatting_prompts_func)
    
    # ---------------------------------------------------------
    # 🚀 ÉTAPE 4 : PARAMÈTRES D'ENTRAÎNEMENT (HYPERPARAMÈTRES)
    # Ces paramètres sont cruciaux pour ne pas faire crasher le GPU (OOM - Out of Memory)
    # ---------------------------------------------------------
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,     # 🔥 Batch size à 1 pour économiser la RAM vidéo
        gradient_accumulation_steps=8,     # On accumule 8 calculs avant de mettre à jour le modèle (Batch effectif = 8)
        gradient_checkpointing=True,       # 🔥 Supprime les données intermédiaires de la RAM pour ne pas exploser la mémoire
        optim="paged_adamw_8bit",          # Optimiseur en 8-bits (divise sa taille mémoire par 4)
        save_steps=50,                     # Sauvegarde un checkpoint toutes les 50 étapes
        logging_steps=10,                  # Affiche l'erreur (Loss) toutes les 10 étapes
        learning_rate=2e-4,                # Vitesse d'apprentissage
        bf16=True,                         # Format Bfloat16 (ultra rapide sur les cartes NVIDIA récentes)
        max_grad_norm=0.3,
        max_steps=200,                     # Nombre total d'étapes (ajustable selon le temps disponible)
        warmup_steps=10,                   # Chauffe doucement le modèle au début
        lr_scheduler_type="constant"
    )
    
    # Lancement du dresseur
    trainer = SFTTrainer(
        model=model, train_dataset=dataset, peft_config=peft_config,
        processing_class=tokenizer, args=training_args
    )
    
    print("Starting training...")
    trainer.train()
    
    # ---------------------------------------------------------
    # 💾 ÉTAPE 5 : SAUVEGARDE DE L'ADAPTATEUR
    # Le modèle de base (14 Go) n'est PAS sauvegardé.
    # On sauvegarde UNIQUEMENT les poids LoRA (40 Mo).
    # ---------------------------------------------------------
    print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
