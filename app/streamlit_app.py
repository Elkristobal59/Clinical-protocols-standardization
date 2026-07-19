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
    from scripts.live_scraper import run_scraper
except ImportError:
    st.error("Impossible d'importer le scraper. Assurez-vous que scripts/live_scraper.py existe.")
    run_scraper = None

st.set_page_config(page_title="Essais Cliniques IA", page_icon="🫀", layout="wide")

st.title("🫀 Moteur d'Extraction & Chatbot Clinique")
st.markdown("Architecture de bout en bout avec GPU, Supabase, et MLflow.")

# Initialisation de l'état
if "extracted_docs" not in st.session_state:
    st.session_state.extracted_docs = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
st.sidebar.header("Architecture Phase 4")
st.sidebar.metric(label="Serveur Inférence", value="Lightning AI (GPU)")
st.sidebar.metric(label="Modèles", value="Qwen-1.5B + BioBERT")
st.sidebar.info("Cette version héberge Streamlit et le scraping, et délègue l'intelligence (Extraction & RAG) au GPU via l'API.")
api_url = st.sidebar.text_input("URL de l'API Lightning AI:", value=os.getenv("LIGHTNING_AI_API_URL", "http://127.0.0.1:8000"), key="api_url_input")

if st.session_state.api_url_input:
    os.environ["LIGHTNING_AI_API_URL"] = st.session_state.api_url_input

tab1, tab2 = st.tabs(["📄 Extraction & Ingestion", "💬 Chatbot RAG"])

with tab1:
    st.header("1. Ingestion & Extraction")
    query = st.text_input("Quelle maladie investiguer ? (ex: Cardiology, Breast Cancer)")
    max_results = st.slider("Nombre d'essais cliniques à extraire :", min_value=1, max_value=10, value=2)
    
    if st.button("Lancer l'Extraction Complète"):
        if query and run_scraper:
            query = query.strip()
            
            with st.spinner("Étape 1/2 : Scraping furtif des serveurs cliniques en cours (Playwright)..."):
                start_time = time.time()
                output_dir = run_scraper(query, max_results=max_results)
                st.success(f"✅ Scraping terminé en {time.time() - start_time:.1f}s")
                
            pdfs = glob.glob(os.path.join(output_dir, "*.pdf"))[:max_results]
            
            if not pdfs:
                st.warning("Aucun PDF trouvé lors du scraping.")
            else:
                with st.spinner(f"Étape 2/2 : Envoi de {len(pdfs)} PDF(s) au serveur GPU (BioBERT + Qwen) & MLflow..."):
                    start_time = time.time()
                    results = []
                    for pdf in pdfs:
                        try:
                            with open(pdf, "rb") as f:
                                response = requests.post(
                                    f"{api_url}/process_pdf",
                                    files={"file": (os.path.basename(pdf), f, "application/pdf")},
                                    data={"disease": query},
                                    headers={"Bypass-Tunnel-Reminder": "true"}
                                )
                            if response.status_code == 200:
                                data = response.json()
                                try:
                                    data["extraction"] = json.loads(data["extraction"])
                                except:
                                    pass
                                results.append(data)
                                st.session_state.extracted_docs.append(data["document"])
                            else:
                                st.error(f"Erreur API ({response.status_code}): {response.text}")
                        except Exception as e:
                            st.error(f"Impossible de se connecter au Serveur GPU : {e}")
                    st.success(f"✅ Inférence GPU terminée en {time.time() - start_time:.1f}s")
                
                # Affichage
                if results:
                    st.session_state.extracted_docs = list(set(st.session_state.extracted_docs)) # Deduplicate
                    for res in results:
                        doc_id = res.get("document", "Document")
                        extraction = res.get("extraction", {})
                        
                        with st.expander(f"📄 Essai: {doc_id} - Pathologie: {res.get('disease')}", expanded=True):
                            if isinstance(extraction, list):
                                # Grouper par type d'entité
                                grouped_entities = {}
                                for ent in extraction:
                                    if isinstance(ent, dict) and "type" in ent and "text" in ent:
                                        ent_type = ent["type"]
                                        ent_text = ent["text"]
                                        if ent_type not in grouped_entities:
                                            grouped_entities[ent_type] = set()
                                        grouped_entities[ent_type].add(ent_text)
                                
                                if not grouped_entities:
                                    st.info("Aucune entité trouvée ou format inattendu.")
                                else:
                                    # Afficher chaque groupe
                                    for ent_type, texts in grouped_entities.items():
                                        st.markdown(f"**{ent_type} :**")
                                        for t in texts:
                                            st.write(f"- {t}")
                            else:
                                st.text(extraction)

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
