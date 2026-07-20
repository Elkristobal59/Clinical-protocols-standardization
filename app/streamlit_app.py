import os
import sys
import time
import requests
import glob
import json
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

# Ajouter le dossier courant au PATH pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from scripts.live_scraper import run_scraper, download_pdf_for_nctid
except ImportError:
    st.error("Impossible d'importer le scraper. Assurez-vous que scripts/live_scraper.py existe.")
    run_scraper = None
    download_pdf_for_nctid = None

st.set_page_config(page_title="Essais Cliniques IA", page_icon="🫀", layout="wide")

# Image d'en-tête (Bannière médicale tech)
banner_path = os.path.join(os.path.dirname(__file__), "assets", "dashboard_medical.jpg")
if os.path.exists(banner_path):
    _, col_img, _ = st.columns([1, 2, 1])
    with col_img:
        st.image(banner_path, use_container_width=True)

st.title("🫀 Moteur d'Extraction & Chatbot Clinique")
st.markdown("### Architecture de bout en bout (Pipeline Full BioBERT)")
st.markdown("---")

# Initialisation de l'état
if "extracted_docs" not in st.session_state:
    st.session_state.extracted_docs = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar - Informations Projet
st.sidebar.title("Clinical Protocols Standardization")
st.sidebar.markdown("**Projet Jedha - Bootcamp AIFS01**")

st.sidebar.markdown("---")
st.sidebar.subheader("👨‍💻 L'Équipe")
st.sidebar.markdown("Patrick Mouliom, Christopher Gilleron, Jérémie Becker, Arnaud Hoarau")

st.sidebar.markdown("---")
with st.sidebar.expander("📌 Objectifs & Contexte", expanded=False):
    st.markdown("""
    **Objectifs :** Convertir les protocoles d'essais cliniques sous forme de texte libre en jeux de données standardisés et structurés.
    
    Le projet se concentre sur la conception et l'implémentation de pipelines basés sur l'IA. 
    
    **Tâches techniques :** Création de workflows d'extraction multi-étapes avec des LLM, décomposition du contenu en concepts cliniques atomiques, recherche sémantique, plongements vectoriels (embeddings), et mapping vers des taxonomies médicales contrôlées. 
    
    **Tâches fonctionnelles :** Support de la curation de données cliniques, amélioration de la cohérence et de la qualité des informations extraites, et création de jeux de données prêts pour l'analyse (conception d'essais cliniques et optimisation de protocoles).
    """)

with st.sidebar.expander("⚙️ Architecture (Backbone)", expanded=False):
    st.markdown("""
    Extraction d’entités standardisées (NER) de protocoles cliniques à partir d’un corpus (taxonomies médicales = Chia) appris par un modèle **BioBERT** :
    - **Entrées :** protocoles (JSON), modèle NLP BioBERT
    - **Sorties :** JSON avec les entités extraites et leurs relations
    - **Objectif métier :** Structurer les données cliniques brutes pour permettre à un assistant conversationnel (RAG) d'interroger et de cibler précisément les informations pertinentes (maladie, traitement, critères d'inclusion).
    """)

with st.sidebar.expander("🚀 Extensions (Nice to have)", expanded=False):
    st.markdown("""
    - Faire de la recherche d’information sur un ou plusieurs protocoles avec du **RAG** (Retrieval-Augmented Generation).
    - Comparer les performances entre LLM concurrents.
    - Analyser les protocoles associés à différents domaines/pathologies.
    - Fine-tuning si suffisamment de données sont disponibles.
    """)

st.sidebar.markdown("---")
st.sidebar.header("Configuration API")
st.sidebar.metric(label="Serveur Inférence", value="Lightning AI (GPU)")
st.sidebar.metric(label="Modèle", value="Full BioBERT")
st.sidebar.info("Cette interface héberge Streamlit et le scraping, et délègue l'intelligence (Extraction & RAG) au GPU via l'API.")
api_url = st.sidebar.text_input("URL de l'API Lightning AI:", value=os.getenv("LIGHTNING_AI_API_URL", "http://127.0.0.1:8000"), key="api_url_input")

if st.session_state.api_url_input:
    os.environ["LIGHTNING_AI_API_URL"] = st.session_state.api_url_input

tab1, tab2 = st.tabs(["📄 Extraction & Ingestion", "💬 Chatbot RAG"])

with tab1:
    st.header("1. Ingestion & Extraction")
    query = st.text_input("Quelle maladie investiguer ? (ex: Breast Cancer, Alzheimer's, Melanoma, Lupus, Cardiology...)")
    max_results = st.slider("Nombre d'essais cliniques à extraire :", min_value=1, max_value=10, value=2)
    force_pdf = st.checkbox("📥 Ignorer le texte natif et forcer le scraping PDF (Plan B)")
    
    if st.button("Lancer l'Extraction Complète"):
        if query and run_scraper:
            query = query.strip()
            
            with st.spinner("Étape 1/2 : Récupération des données cliniques (JSON natif et/ou PDF)..."):
                start_time = time.time()
                output_dir = os.path.abspath(f"data/live_pdfs_{query.replace(' ', '_')}")
                os.makedirs(output_dir, exist_ok=True)
                
                tasks = []
                # Appeler l'API officielle
                if force_pdf:
                    # Si on force le PDF, il faut chercher des essais qui ont VRAIMENT un PDF attaché
                    api_url_ct = f"https://clinicaltrials.gov/api/v2/studies?query.cond={query}&pageSize=50&fields=NCTId,ProtocolSection,DocumentSection"
                    try:
                        resp = requests.get(api_url_ct, timeout=10)
                        studies = resp.json().get("studies", [])
                        for study in studies:
                            nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                            docs = study.get("documentSection", {}).get("largeDocumentModule", {}).get("largeDocs", [])
                            has_pdf = any(str(d.get("filename", "")).lower().endswith(".pdf") for d in docs)
                            if has_pdf:
                                tasks.append({"type": "pdf", "nct_id": nct_id})
                            if len(tasks) >= max_results:
                                break
                    except Exception as e:
                        st.error(f"Erreur API ClinicalTrials: {e}")
                else:
                    # Recherche standard (priorité au texte)
                    api_url_ct = f"https://clinicaltrials.gov/api/v2/studies?query.cond={query}&pageSize={max_results}&fields=NCTId,ProtocolSection"
                    try:
                        resp = requests.get(api_url_ct, timeout=10)
                        studies = resp.json().get("studies", [])
                        for study in studies:
                            nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                            try:
                                eligibility = study["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
                                if len(eligibility) > 100:
                                    tasks.append({"type": "text", "nct_id": nct_id, "text": eligibility})
                                    continue
                            except KeyError:
                                pass
                            # Fallback sur le PDF si pas de texte (rare, et échouera s'il n'y a pas de PDF)
                            tasks.append({"type": "pdf", "nct_id": nct_id})
                    except Exception as e:
                        st.error(f"Erreur API ClinicalTrials: {e}")
                
                st.success(f"✅ Données récupérées (ou identifiées) en {time.time() - start_time:.1f}s")
                
            if not tasks:
                st.warning("Aucun essai trouvé pour cette requête.")
            else:
                with st.spinner(f"Étape 2/2 : Envoi de {len(tasks)} protocoles au serveur GPU (Full BioBERT) & MLflow..."):
                    start_time = time.time()
                    results = []
                    
                    for task in tasks:
                        nct_id = task["nct_id"]
                        try:
                            if task["type"] == "text":
                                # Mode rapide : Texte direct
                                response = requests.post(
                                    f"{api_url}/process_text",
                                    data={"disease": query, "document_id": nct_id, "text_content": task["text"]},
                                    headers={"Bypass-Tunnel-Reminder": "true"}
                                )
                            else:
                                # Mode fallback : Téléchargement PDF puis envoi
                                pdf_path = download_pdf_for_nctid(nct_id, output_dir) if download_pdf_for_nctid else None
                                if pdf_path and os.path.exists(pdf_path):
                                    with open(pdf_path, "rb") as f:
                                        response = requests.post(
                                            f"{api_url}/process_pdf",
                                            files={"file": (f"{nct_id}.pdf", f, "application/pdf")},
                                            data={"disease": query},
                                            headers={"Bypass-Tunnel-Reminder": "true"}
                                        )
                                else:
                                    st.warning(f"Impossible de trouver le texte ou le PDF pour l'essai {nct_id}")
                                    continue
                                    
                            if response.status_code == 200:
                                data = response.json()
                                try:
                                    raw_text = data["extraction"].strip()
                                    # Nettoyage des backticks markdown (ex: ```json ... ```)
                                    import re
                                    json_match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
                                    if json_match:
                                        raw_text = json_match.group(1)
                                    
                                    cleaned_ext = raw_text.replace('"\n"', '",\n"').replace(']\n"', '],\n"').replace('}\n"', '},\n"')
                                    data["extraction"] = json.loads(cleaned_ext)
                                except:
                                    try:
                                        data["extraction"] = json.loads(raw_text)
                                    except:
                                        data["extraction"] = {"erreur": "Le LLM n'a pas retourné un JSON valide", "raw_output": raw_text}
                                results.append(data)
                                st.session_state.extracted_docs.append(data["document"])
                            else:
                                st.error(f"Erreur API ({response.status_code}) pour {nct_id}: {response.text}")
                        except Exception as e:
                            st.error(f"Impossible de traiter l'essai {nct_id} : {e}")
                            
                    st.success(f"✅ Inférence GPU terminée en {time.time() - start_time:.1f}s")
                
                # Sauvegarde en mémoire pour survivre au changement d'onglet
                st.session_state.latest_results = results
                st.session_state.latest_query = query
                
    # Affichage (En dehors du bouton pour rester visible)
    if getattr(st.session_state, 'latest_results', []):
        results = st.session_state.latest_results
        query = st.session_state.latest_query
        
        st.session_state.extracted_docs = list(set(st.session_state.extracted_docs)) # Deduplicate
        
        flattened_data = []
        for res in results:
            doc_id = res['document']
            disease = res.get('disease', 'N/A')
            ext = res.get("extraction", {})
            
            with st.expander(f"📄 Essai: {doc_id} - Pathologie: {disease}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**🌐 Source Officielle :** [Voir sur ClinicalTrials.gov](https://clinicaltrials.gov/study/{doc_id})")
                with col2:
                    supabase_url = os.getenv("SUPABASE_API_URL", "").rstrip("/")
                    if supabase_url:
                        pdf_link = f"{supabase_url}/storage/v1/object/public/clinical_pdfs/{doc_id}.pdf"
                        st.markdown(f"**📥 Archive PDF :** [Télécharger depuis Supabase]({pdf_link}) *(Uniquement si scrapé en Plan B)*")
                st.json(ext)
            # Extraction robuste pour gérer les listes de dictionnaires OU les listes de strings
            meds_list = ext.get("medications", []) if isinstance(ext, dict) else []
            meds = ", ".join([str(m.get("name", m.get("description", ""))) if isinstance(m, dict) else str(m) for m in meds_list])
            
            criteria_list = ext.get("inclusion_criteria", []) if isinstance(ext, dict) else []
            criteria = ", ".join([str(c.get("description", c.get("category", ""))) if isinstance(c, dict) else str(c) for c in criteria_list])
            
            condition = ext.get("condition", "") if isinstance(ext, dict) else ""
            
            flattened_data.append({
                "Document": doc_id,
                "Pathologie": disease,
                "Condition Principale": condition,
                "Médicaments (Drug)": meds,
                "Critères (Measurement)": criteria
            })
        
        st.markdown("---")
        st.subheader("📊 Tableau de Bord Clinique")
        import pandas as pd
        df = pd.DataFrame(flattened_data)
        st.dataframe(df, use_container_width=True)
        
        # Fix: Utilisation de utf-8-sig pour que Excel lise bien les accents
        csv = df.to_csv(index=False, sep=';').encode('utf-8-sig')
        st.download_button(
            label="📥 Exporter les extractions (Excel/CSV)",
            data=csv,
            file_name=f"extractions_{query.replace(' ', '_')}.csv",
            mime="text/csv"
        )

with tab2:
    st.header("2. Assistant Chatbot RAG")
    st.markdown("Posez vos questions sur les essais cliniques indexés en base.")
    
    doc_filter = st.selectbox("Filtrer par essai clinique (Optionnel) :", ["Toute la base"] + st.session_state.extracted_docs)
    
    # Affichage de l'historique
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "context" in msg:
                with st.expander("🔍 Voir les sources"):
                    for c in msg["context"]:
                        st.caption(c)

    # Zone de saisie
    if prompt := st.chat_input("Posez votre question clinique..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🧠 Réflexion en cours (Recherche vectorielle + Génération LLM)...")
            
            try:
                selected_doc = None if doc_filter == "Toute la base" else doc_filter
                
                response = requests.post(
                    f"{api_url}/chat_rag",
                    data={"question": prompt, "doc_id": selected_doc},
                    headers={"Bypass-Tunnel-Reminder": "true"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Erreur de génération.")
                    context = data.get("context", [])
                    
                    message_placeholder.markdown(answer)
                    with st.expander("🔍 Voir les sources utilisées"):
                        for c in context:
                            st.caption(c)
                            
                    st.session_state.chat_history.append({"role": "assistant", "content": answer, "context": context})
                else:
                    message_placeholder.error(f"Erreur API ({response.status_code}): {response.text}")
            except Exception as e:
                message_placeholder.error(f"Impossible de joindre l'API : {e}")
