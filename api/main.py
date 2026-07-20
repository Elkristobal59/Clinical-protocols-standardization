import os
import io
import fitz  # PyMuPDF
import psycopg2
import torch
import time
import mlflow
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
try:
    from vllm import LLM, SamplingParams
except ImportError:
    pass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Clinical Trials AI Extraction API (Phase 3)")

SUPABASE_DB_URL = os.getenv("SUPABASE_DATABASE_URL")
BIOBERT_MODEL = "dmis-lab/biobert-v1.1"
QWEN_MODEL = "Qwen/Qwen1.5-1.8B-Chat"

# Variables globales pour stocker les modèles
device = None
biobert_tokenizer = None
biobert_model = None
biobert_ner_pipeline = None
qwen_tokenizer = None
qwen_model = None
conn = None
is_vllm = False

@app.on_event("startup")
async def startup_event():
    """Charge les modèles lourds en RAM/VRAM une seule fois au démarrage."""
    global device, biobert_tokenizer, biobert_model, biobert_ner_pipeline, qwen_tokenizer, qwen_model, conn, is_vllm
    
    print("Initialisation du pipeline sur GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Chargement {BIOBERT_MODEL} (Embeddings)...")
    biobert_tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
    biobert_model = AutoModel.from_pretrained(BIOBERT_MODEL).to(device)
    
    print("Chargement BioBERT-NER (Extraction)...")
    from transformers import pipeline
    biobert_ner_pipeline = pipeline(
        "token-classification",
        model="docs/02_benchmark/biobert-chia-ner",
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    
    print(f"Chargement {QWEN_MODEL} (LLM Génératif)...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    
    # On n'active vLLM que si on a un GPU (vLLM plante sur CPU pur)
    if torch.cuda.is_available():
        try:
            print("GPU détecté -> Activation de vLLM...")
            qwen_model = LLM(model=QWEN_MODEL, gpu_memory_utilization=0.7)
            is_vllm = True
        except Exception as e:
            print(f"Erreur vLLM, fallback transformers: {e}")
            is_vllm = False
            
    if not is_vllm:
        print("Mode CPU ou Fallback -> Activation de Transformers...")
        qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
    
    print("Connexion Supabase...")
    conn = psycopg2.connect(SUPABASE_DB_URL)
    
    print("Configuration MLflow...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Clinical_Trials_Extraction")
    
    print("✅ Serveur Prêt.")

def get_biobert_embedding(text):
    """Retourne l'embedding BioBERT (768 dimensions) pour un texte donné."""
    encoded_input = biobert_tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        out = biobert_model(**encoded_input)
    # Mean pooling
    emb = torch.sum(out[0] * encoded_input['attention_mask'].unsqueeze(-1), 1) / torch.clamp(encoded_input['attention_mask'].sum(1, keepdim=True), min=1e-9)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb[0].cpu().tolist()

@app.post("/process_pdf")
async def process_pdf(
    file: UploadFile = File(...),
    disease: str = Form(...)
):
    """
    Route principale qui :
    1. Reçoit le PDF.
    2. L'encode dans Supabase via BioBERT.
    3. Fait l'extraction sémantique via Qwen.
    """
    cur = None
    start_time = time.time()
    try:
        # --- 1. LECTURE DU PDF EN MEMOIRE ---
        content = await file.read()
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            text += doc.load_page(page_num).get_text() + "\n"
        doc.close()
        
        filename = file.filename.replace(".pdf", "")
        print(f"PDF {filename} reçu, extraction de texte terminée.")
        
        # --- 2. CHUNKING & INGESTION (BioBERT) ---
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)
        
        # Rollback de toute transaction cassée avant de commencer
        conn.rollback()
        cur = conn.cursor()
        
        for idx, chunk_text in enumerate(chunks):
            embedding = get_biobert_embedding(chunk_text)
            cur.execute("""
                INSERT INTO clinical_trials_data_biobert (doc_id, chunk_id, raw_text, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO NOTHING
            """, (filename, f"{filename}_chunk_{idx}", chunk_text, embedding))
        conn.commit()
        print(f"Ingestion terminée pour {filename}.")
        
        # --- 3. RECHERCHE SEMANTIQUE (MAP) ---
        query = f"inclusion criteria medications {disease}"
        query_emb = get_biobert_embedding(query)
        
        # On limite la recherche spécifiquement au document qu'on vient d'uploader
        cur.execute("""
            SELECT raw_text
            FROM clinical_trials_data_biobert
            WHERE doc_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT 5;
        """, (filename, query_emb))
        
        results = cur.fetchall()
        cur.close()
        cur = None
        
        if not results:
            raise HTTPException(status_code=404, detail="Aucun texte trouvé pour ce document.")
            
        # --- 4. EXTRACTION (REDUCE - BioBERT NER) ---
        print(f"Extraction BioBERT-NER pour {disease}...")
        
        CONTENT = ["Condition", "Drug", "Procedure", "Measurement", "Value", "Temporal", "Observation", "Person", "Device"]
        extraction = []
        seen = set()
        
        for r in results:
            seg = r[0]
            for e in biobert_ner_pipeline(seg):
                ty = e.get("entity_group")
                w = e.get("word", "").strip()
                if ty in CONTENT and w and (w.lower(), ty) not in seen:
                    seen.add((w.lower(), ty))
                    extraction.append({"text": w, "type": ty})
                    
        response_json = extraction
        
        latency = time.time() - start_time
        
        # Tracking MLflow
        with mlflow.start_run():
            mlflow.log_param("disease", disease)
            mlflow.log_param("document", filename)
            mlflow.log_param("model", "biobert-chia-ner")
            mlflow.log_metric("latency_sec", latency)
            import json
            mlflow.log_text(json.dumps(response_json, ensure_ascii=False), "response.json")
            
        return {"status": "success", "disease": disease, "document": filename, "extraction": response_json}

    except HTTPException:
        raise  # Laisser passer les erreurs HTTP (404, etc.)
    except Exception as e:
        # Rollback pour remettre la connexion dans un état propre
        try:
            conn.rollback()
        except:
            pass
        print(f"Erreur API: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur is not None:
            try:
                cur.close()
            except:
                pass

@app.post("/chat_rag")
async def chat_rag(question: str = Form(...), doc_id: str = Form(None)):
    """
    Route pour le Chatbot RAG. 
    Cherche dans Supabase puis génère une réponse.
    """
    cur = None
    start_time = time.time()
    try:
        query_emb = get_biobert_embedding(question)
        
        conn.rollback()
        cur = conn.cursor()
        
        if doc_id:
            cur.execute("""
                SELECT raw_text FROM clinical_trials_data_biobert
                WHERE doc_id = %s
                ORDER BY embedding <=> %s::vector LIMIT 5;
            """, (doc_id, query_emb))
        else:
            cur.execute("""
                SELECT raw_text FROM clinical_trials_data_biobert
                ORDER BY embedding <=> %s::vector LIMIT 5;
            """, (query_emb,))
            
        results = cur.fetchall()
        cur.close()
        cur = None
        
        if not results:
            return {"answer": "Je n'ai pas trouvé d'informations pertinentes dans la base."}
            
        context = "\n\n".join([f"Extrait:\n{r[0]}" for r in results])
        
        prompt = f"""You are a helpful medical assistant. Answer the user's question based ONLY on the following context extracts. If you don't know the answer based on the context, say so. Answer in French.

Context Extracts:
{context}

Question: {question}
Answer:"""

        messages = [
            {"role": "system", "content": "You are a helpful medical AI."},
            {"role": "user", "content": prompt}
        ]
        
        text_prompt = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        print(f"Génération RAG pour: {question}...")
        
        if is_vllm:
            # vLLM path
            sampling_params = SamplingParams(temperature=0.3, max_tokens=512)
            outputs = qwen_model.generate([text_prompt], sampling_params)
            answer = outputs[0].outputs[0].text
        else:
            # Fallback transformers path
            model_inputs = qwen_tokenizer([text_prompt], return_tensors="pt").to(device)
            with torch.no_grad():
                generated_ids = qwen_model.generate(model_inputs.input_ids, max_new_tokens=512, temperature=0.3)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            answer = qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        latency = time.time() - start_time
        with mlflow.start_run():
            mlflow.log_param("task", "chat_rag")
            mlflow.log_param("doc_id", doc_id)
            mlflow.log_param("question", question)
            mlflow.log_metric("latency_sec", latency)
            mlflow.log_text(prompt, "rag_prompt.txt")
            mlflow.log_text(answer, "rag_response.txt")
            
        return {"answer": answer, "context": [r[0] for r in results]}

    except Exception as e:
        try: conn.rollback()
        except: pass
        print(f"Erreur RAG API: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur is not None:
            try: cur.close()
            except: pass
