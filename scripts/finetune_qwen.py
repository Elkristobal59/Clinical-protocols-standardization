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

# Configuration
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" # Modèle de 0.5B pour garantir le passage sur 6 Go VRAM
DATASET_PATH = "data/train_dataset.jsonl"
OUTPUT_DIR = r"d:\AIFS01\PROJET FINAL\stack_equipe\models\qwen_chia_finetuned"

def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # We will use 4-bit quantization (QLoRA) to save VRAM (fits on RTX 3060 Laptop 6GB)
    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"Loading model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Format the messages into prompt
    def formatting_prompts_func(example):
        messages = example["messages"]
        # Apply the conversational chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}
        
    dataset = dataset.map(formatting_prompts_func)
    
    # Training Arguments optimisés pour 6 Go VRAM (Laptop)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_seq_length=512,
        per_device_train_batch_size=1,     # Réduit à 1 pour économiser la mémoire
        gradient_accumulation_steps=8,     # Augmenté pour simuler un batch size de 8
        optim="paged_adamw_8bit",          # Optimiseur en 8-bits pour diviser sa taille mémoire par 4 !
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=200, 
        warmup_steps=10,
        lr_scheduler_type="constant"
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model, train_dataset=dataset, peft_config=peft_config,
        processing_class=tokenizer, args=training_args
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
