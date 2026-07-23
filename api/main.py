"""
Script : main.py (Le Cœur du Réacteur - Backend API)
----------------------------------------------------
Rôle : Exposer les fonctionnalités d'IA (RAG, Extraction) via des points d'accès HTTP (FastAPI) 
pour que le front-end (Streamlit) puisse les interroger.

🎓 Explication pour le jury (L'Architecture Hybride) :
Ce script est la définition même du MLOps. Il combine :
1. FastAPI : Un serveur web ultra-rapide asynchrone.
2. Transformers / vLLM : Pour charger les modèles d'IA en mémoire vidéo (VRAM).
3. LangChain : Pour découper intelligemment les textes (Chunking).
4. pgvector (Supabase) : La base de données vectorielle pour la recherche sémantique.
5. MLflow : Pour monitorer en direct (tracking) les prompts, les réponses et la latence.

Point fort : Nous utilisons `vLLM` si un GPU puissant est détecté. C'est un moteur d'inférence 
qui gère la VRAM avec la technique "PagedAttention", rendant la génération de texte 2 à 5 fois plus rapide !
"""

import os
import io
import fitz  # PyMuPDF : Pour lire les PDF directement en mémoire sans les sauvegarder
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

# ---------------------------------------------------------
# ⚙️ CONFIGURATION DES MODÈLES ET SERVICES
# ---------------------------------------------------------
SUPABASE_DB_URL = os.getenv("SUPABASE_DATABASE_URL")
BIOBERT_MODEL = "dmis-lab/biobert-v1.1" # Modèle de NLP médical pour la recherche vectorielle (Retriever)
QWEN_MODEL = "Qwen/Qwen1.5-7B-Chat"  # LLM Génératif pour extraire le JSON (Generator)

# Variables globales : On charge les modèles une seule fois au démarrage pour éviter de saturer la RAM.
device = None
biobert_tokenizer = None
biobert_model = None
qwen_tokenizer = None
qwen_model = None
conn = None
is_vllm = False

@app.on_event("startup")
async def startup_event():
    """
    S'exécute automatiquement au lancement du serveur FastAPI (`uvicorn api.main:app`).
    Charge les modèles lourds en RAM/VRAM une seule fois (Singleton pattern).
    """
    global device, biobert_tokenizer, biobert_model, qwen_tokenizer, qwen_model, conn, is_vllm
    
    print("Initialisation du pipeline sur GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 📥 CHARGEMENT DE BIOBERT (Modèle léger)
    print(f"Chargement {BIOBERT_MODEL} (Embeddings)...")
    biobert_tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
    biobert_model = AutoModel.from_pretrained(BIOBERT_MODEL).to(device)
    
    # 📥 CHARGEMENT DE QWEN (Modèle Lourd - 7 Milliards de paramètres)
    print(f"Chargement {QWEN_MODEL} (LLM Génératif)...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    
    # 🚀 OPTIMISATION vLLM (Si GPU disponible)
    # vLLM est une technologie d'inférence de pointe pour les LLMs. 
    # Elle évite la fragmentation de la VRAM (PagedAttention).
    if torch.cuda.is_available():
        try:
            print("GPU détecté -> Activation de vLLM...")
            # On réserve 85% de la VRAM au LLM, et on limite la fenêtre de contexte (tokens) pour éviter les crashs OOM.
            qwen_model = LLM(model=QWEN_MODEL, gpu_memory_utilization=0.85, max_model_len=4096)
            is_vllm = True
        except Exception as e:
            print(f"Erreur vLLM, fallback transformers: {e}")
            is_vllm = False
            
    # 🐌 FALLBACK CLASSIC TRANSFORMERS (Si pas de GPU ou erreur vLLM)
    if not is_vllm:
        print("Mode CPU ou Fallback -> Activation de Transformers...")
        qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
    
    print("Connexion Supabase (Base Vectorielle)...")
    conn = psycopg2.connect(SUPABASE_DB_URL)
    
    # 📊 MONITORING MLOPS AVEC MLFLOW
    print("Configuration MLflow...")
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlflow.db"))
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Clinical_Trials_Extraction") # Tous nos logs iront dans cette "boîte"
    
    print("✅ Serveur Prêt. L'API est en ligne !")

def get_biobert_embedding(text):
    """
    Transforme un texte médical (ex: "Patient adulte avec diabète") en 
    Vecteur mathématique de 768 dimensions grâce à BioBERT.
    Ce vecteur permettra de faire une recherche sémantique (similarité cosinus).
    """
    encoded_input = biobert_tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        out = biobert_model(**encoded_input)
    # "Mean pooling" : On fait la moyenne des vecteurs de chaque mot pour obtenir LE vecteur de la phrase entière
    emb = torch.sum(out[0] * encoded_input['attention_mask'].unsqueeze(-1), 1) / torch.clamp(encoded_input['attention_mask'].sum(1, keepdim=True), min=1e-9)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1) # Normalisation pour la similarité cosinus (Vector Database)
    return emb[0].cpu().tolist()

def process_extracted_text(text: str, filename: str, disease: str, start_time: float) -> dict:
    """
    Le cœur du Pipeline RAG (Retrieval-Augmented Generation).
    1. Découpe le texte (LangChain).
    2. Vectorise et stocke (BioBERT -> Supabase pgvector).
    3. Recherche les paragraphes les plus pertinents (Top 5).
    4. Envoie ces 5 paragraphes au LLM (Qwen) pour extraire le JSON final.
    """
    global conn, cur, biobert_model, biobert_tokenizer, qwen_model, qwen_tokenizer, device, is_vllm
    cur = None
    try:
        # --- 1. CHUNKING (Découpage) ---
        # On ne peut pas donner un PDF de 100 pages à l'IA d'un coup (la mémoire exploserait).
        # On le coupe en morceaux de 1000 caractères, avec un chevauchement de 200 pour ne pas couper une phrase au milieu.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)
        
        # Rollback de toute transaction SQL cassée avant de commencer
        conn.rollback()
        cur = conn.cursor()
        
        # --- 2. INGESTION VECTORIELLE ---
        for idx, chunk_text in enumerate(chunks):
            embedding = get_biobert_embedding(chunk_text)
            cur.execute("""
                INSERT INTO clinical_trials_data_biobert (doc_id, chunk_id, raw_text, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO NOTHING
            """, (filename, f"{filename}_chunk_{idx}", chunk_text, embedding))
        conn.commit()
        print(f"Ingestion terminée pour {filename}.")
        
        # --- 3. RECHERCHE SEMANTIQUE (MAP / RETRIEVAL) ---
        query = f"inclusion criteria medications {disease}"
        query_emb = get_biobert_embedding(query)
        
        # Le "ORDER BY embedding <=> %s::vector" est la syntaxe magique de l'extension `pgvector` de Postgres
        # pour trouver instantanément les vecteurs les plus proches mathématiquement.
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
            
        context = "\n\n".join([f"Extrait:\n{r[0]}" for r in results])
        
        # --- 4. EXTRACTION JSON (REDUCE / GENERATOR) ---
        prompt = f"""You are a clinical trials expert. Extract the medications and patient inclusion criteria for {disease} from the following text extracts.
Format your answer in clear JSON format: {{"condition": "...", "medications": [...], "inclusion_criteria": "..."}}

Context Extracts:
{context}

Response (JSON only):
"""
        messages = [
            {"role": "system", "content": "You are a helpful and precise medical AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text_prompt = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        print(f"Génération Qwen pour {disease}...")
        
        if is_vllm:
            # 🚀 Inférence hyper rapide via vLLM
            sampling_params = SamplingParams(temperature=0.1, max_tokens=2048, repetition_penalty=1.1)
            outputs = qwen_model.generate([text_prompt], sampling_params)
            response_json = outputs[0].outputs[0].text
        else:
            # 🐌 Inférence standard (HuggingFace Transformers)
            model_inputs = qwen_tokenizer([text_prompt], return_tensors="pt").to(device)
            with torch.no_grad():
                generated_ids = qwen_model.generate(model_inputs.input_ids, max_new_tokens=2048, temperature=0.1, repetition_penalty=1.1)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response_json = qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        latency = time.time() - start_time
        
        # --- 5. LOGGING MLOPS (L'Oeil de Sauron) ---
        # MLflow va enregistrer la vitesse de réponse (latency), le prompt utilisé, et le JSON généré.
        # Cela permet à l'équipe Data Science de surveiller les "hallucinations" ou les baisses de perf.
        with mlflow.start_run():
            mlflow.log_param("disease", disease)
            mlflow.log_param("document", filename)
            mlflow.log_param("model", QWEN_MODEL)
            mlflow.log_metric("latency_sec", latency)
            mlflow.log_text(prompt, "prompt.txt")
            mlflow.log_text(response_json, "response.json")
            
        return {"status": "success", "disease": disease, "document": filename, "extraction": response_json}

    except HTTPException:
        raise  # Laisser passer les erreurs HTTP propres (404, etc.)
    except Exception as e:
        try:
            conn.rollback() # Annule la transaction SQL en cas d'erreur grave
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


@app.post("/process_text")
async def process_text(
    disease: str = Form(...),
    document_id: str = Form(...),
    text_content: str = Form(...)
):
    """
    Route API (Point d'accès) #1 : Traitement direct de texte.
    Optimisation majeure : Si on a déjà récupéré le texte propre depuis l'API ClinicalTrials,
    on esquive toute l'étape d'extraction PyMuPDF !
    """
    start_time = time.time()
    print(f"Réception du texte direct pour l'essai {document_id}")
    return process_extracted_text(text=text_content, filename=document_id, disease=disease, start_time=start_time)


@app.post("/process_pdf")
async def process_pdf(
    file: UploadFile = File(...),
    disease: str = Form(...)
):
    """
    Route API (Point d'accès) #2 : Le Fallback historique (Traitement PDF).
    1. Reçoit le PDF uploadé par l'utilisateur.
    2. Sauvegarde le fichier brut dans Supabase Storage (Archivage).
    3. Lit le PDF en mémoire via PyMuPDF (fitz) et lance le pipeline RAG.
    """
    start_time = time.time()
    try:
        # --- 1. LECTURE DU PDF EN MEMOIRE ---
        # "await file.read()" -> On ne sauvegarde jamais le PDF sur le disque du serveur API. 
        # C'est plus sécurisé et beaucoup plus rapide.
        content = await file.read()
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            text += doc.load_page(page_num).get_text() + "\n"
        doc.close()
        
        filename = file.filename.replace(".pdf", "")
        print(f"PDF {filename} reçu, extraction de texte terminée.")
        
        # --- 2. UPLOAD PDF SUR SUPABASE STORAGE (Archivage Cloud) ---
        supa_url = os.getenv("SUPABASE_API_URL")
        supa_key = os.getenv("SUPABASE_ANON_KEY")
        if supa_url and supa_key:
            try:
                import requests
                headers = {"Authorization": f"Bearer {supa_key}", "apikey": supa_key, "Content-Type": "application/pdf"}
                # Upload dans le bucket sécurisé nommé 'clinical_pdfs'
                res = requests.put(f"{supa_url}/storage/v1/object/clinical_pdfs/{file.filename}", data=content, headers=headers)
                if res.status_code in [200, 201]:
                    print(f"PDF uploadé sur Supabase Storage: {file.filename}")
                else:
                    print(f"Erreur Supabase Storage: {res.text}")
            except Exception as e:
                print(f"Erreur lors de l'upload Supabase: {e}")
                
        # 3. Lancement du RAG
        return process_extracted_text(text=text, filename=filename, disease=disease, start_time=start_time)

    except Exception as e:
        print(f"Erreur lecture PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_rag")
async def chat_rag(question: str = Form(...), doc_id: str = Form(None)):
    """
    Route API (Point d'accès) #3 : Le Chatbot Conversationnel RAG (Retrieval-Augmented Generation).
    Permet à l'utilisateur de poser une question libre en langage naturel à l'IA.
    """
    cur = None
    start_time = time.time()
    try:
        # 1. On vectorise la question de l'utilisateur (BioBERT)
        query_emb = get_biobert_embedding(question)
        
        conn.rollback()
        cur = conn.cursor()
        
        # 2. On interroge Supabase (Recherche des 5 paragraphes les plus pertinents)
        if doc_id:
            # Recherche ciblée sur un essai clinique précis
            cur.execute("""
                SELECT raw_text FROM clinical_trials_data_biobert
                WHERE doc_id = %s
                ORDER BY embedding <=> %s::vector LIMIT 5;
            """, (doc_id, query_emb))
        else:
            # Recherche Globale (sur toute la base documentaire)
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
        
        # 3. Construction du Prompt : On donne le contexte (les extraits) ET la question (User) au LLM
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
        
        # 4. Génération de la réponse (Qwen)
        if is_vllm:
            # vLLM
            sampling_params = SamplingParams(temperature=0.3, max_tokens=512, repetition_penalty=1.1)
            outputs = qwen_model.generate([text_prompt], sampling_params)
            answer = outputs[0].outputs[0].text
        else:
            # Transformers fallback
            model_inputs = qwen_tokenizer([text_prompt], return_tensors="pt").to(device)
            with torch.no_grad():
                generated_ids = qwen_model.generate(model_inputs.input_ids, max_new_tokens=512, temperature=0.3, repetition_penalty=1.1)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            answer = qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        latency = time.time() - start_time
        
        # 5. Monitoring
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
