import os
import sys
import time
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import pandas as pd

# Ajouter le dossier parent au PATH pour les imports (scripts/…)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from scripts.live_scraper import download_pdf_for_nctid
except ImportError:
    download_pdf_for_nctid = None

st.set_page_config(page_title="Essais Cliniques IA", page_icon="🫀", layout="wide")


# --------------------------------------------------------------------------- #
# Summary table : logique autonome (champs de Jérémie)
# Construite à partir du JSON déjà récupéré à la recherche -> AUCUNE 2e requête.
# --------------------------------------------------------------------------- #
def _safe_get(d, *keys):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def study_summary_row(sj):
    ps = "protocolSection"
    inters = _safe_get(sj, ps, "armsInterventionsModule", "interventions") or []
    int_names = " | ".join(i.get("name") for i in inters if i.get("name")) or "N/A"
    int_types = " | ".join(i.get("type") for i in inters if i.get("type")) or "N/A"
    outcomes = _safe_get(sj, ps, "outcomesModule", "primaryOutcomes") or []
    outcome = " | ".join(o.get("measure") for o in outcomes if o.get("measure")) or "N/A"
    phases = _safe_get(sj, ps, "designModule", "phases")
    return {
        "NCT_ID": _safe_get(sj, ps, "identificationModule", "nctId"),
        "Official Title": _safe_get(sj, ps, "identificationModule", "officialTitle") or "N/A",
        "Study Type": _safe_get(sj, ps, "designModule", "studyType") or "N/A",
        "Phase": ", ".join(phases) if phases else "N/A",
        "Primary Purpose": _safe_get(sj, ps, "designModule", "designInfo", "primaryPurpose") or "N/A",
        "Enrollment Count": _safe_get(sj, ps, "designModule", "enrollmentInfo", "count") or "N/A",
        "Eligibility Criteria": "Présent" if _safe_get(sj, ps, "eligibilityModule", "eligibilityCriteria") else "Absent",
        "Intervention Type": int_types,
        "Intervention Name": int_names,
        "Primary Outcome Measure": outcome,
    }

# --------------------------------------------------------------------------- #
# Définition des champs (table de Jérémie)
# --------------------------------------------------------------------------- #
# Champs proposés dans les menus déroulants de recherche (onglet 1)
QUERY_FIELDS = ["Condition", "MinimumAge", "MaximumAge", "Sex", "HealthyVolunteers",
                "InterventionType", "Phase", "StudyType", "DesignPrimaryPurpose"]

# Valeurs fixes -> vrai menu déroulant (moins de 10 options)
ENUM_OPTIONS = {
    "Sex": ["ALL", "FEMALE", "MALE"],
    "HealthyVolunteers": ["Yes", "No"],
    "InterventionType": ["DRUG", "DEVICE", "BIOLOGICAL", "PROCEDURE", "RADIATION",
                          "BEHAVIORAL", "GENETIC", "DIETARY_SUPPLEMENT",
                          "COMBINATION_PRODUCT", "DIAGNOSTIC_TEST", "OTHER"],
    "Phase": ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"],
    "StudyType": ["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"],
    "DesignPrimaryPurpose": ["TREATMENT", "PREVENTION", "DIAGNOSTIC", "SUPPORTIVE_CARE",
                             "SCREENING", "HEALTH_SERVICES_RESEARCH", "BASIC_SCIENCE",
                             "DEVICE_FEASIBILITY", "OTHER"],
}
# Champs du filtre "post-JSON" (onglet 3, appliqué sur les résultats)
FILTER_FIELDS = ["EligibilityCriteria", "PrimaryOutcomeMeasure",
                 "InterventionName", "Phase", "StudyType"]

# Libellés francisés à afficher (comme Christopher : "Maladie (Condition)").
# Les autres champs gardent leur nom API en anglais.
FIELD_LABELS = {"Condition": "Maladie (Condition)"}

# --------------------------------------------------------------------------- #
# État de session
# --------------------------------------------------------------------------- #
for k, v in {"search_done": False, "analysis_done": False,
             "found_studies": [], "force_pdf": False, "latest_query": "",
             "latest_results": [], "extracted_docs": [],
             "chat_history": [], "demo_cache": {}, "selected_ncts": []}.items():
    st.session_state.setdefault(k, v)

# --------------------------------------------------------------------------- #
# En-tête + sidebar
# --------------------------------------------------------------------------- #
banner_path = os.path.join(os.path.dirname(__file__), "assets", "dashboard_medical.jpg")
if os.path.exists(banner_path):
    _, col_img, _ = st.columns([1, 2, 1])
    with col_img:
        st.image(banner_path, use_container_width=True)

st.title("🫀 Moteur d'Extraction & Chatbot Clinique")
st.markdown("### Architecture de bout en bout (ETL Hybride & vLLM)")
st.markdown("---")

st.sidebar.title("Clinical Protocols Standardization")
st.sidebar.markdown("**Projet Jedha - Bootcamp AIFS01**")
st.sidebar.markdown("---")
st.sidebar.subheader("👨‍💻 L'Équipe")
st.sidebar.markdown("Patrick Mouliom, Christopher Gilleron, Jérémie Becker, Arnaud Hoarau")
st.sidebar.markdown("---")
st.sidebar.header("Configuration API")
st.sidebar.metric(label="Serveur Inférence", value="Lightning AI (GPU)")
st.sidebar.metric(label="Moteur Inférence", value="vLLM + Qwen")
api_url = st.sidebar.text_input(
    "URL de l'API Lightning AI:",
    value=os.getenv("LIGHTNING_AI_API_URL", "http://127.0.0.1:8000"),
    key="api_url_input")
if st.session_state.api_url_input:
    os.environ["LIGHTNING_AI_API_URL"] = st.session_state.api_url_input


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def lock_need_search():
    """Onglet 2 : tant qu'aucune recherche n'a été lancée."""
    st.info("🔒 Lancez d'abord une recherche dans l'onglet « 🔎 Critères de sélection ».")


def lock_need_analysis():
    """Onglets 3 & 4 : tant que la Summary table n'a pas généré l'extraction GPU."""
    st.info("🔒 Cochez des études dans « 📑 Summary table » puis cliquez "
            "« 🚀 Analyser (GPU) » pour débloquer cet onglet.")


def build_api_query(values):
    """Construit l'URL de requête ClinicalTrials v2 à partir des critères choisis.
    - Condition -> query.cond (recherche textuelle)
    - champs à valeurs fixes -> filter.advanced (syntaxe Essie AREA[...])
    - Age -> géré en post-filtre (voir age_ok)
    Retourne (query_string, label_lisible)."""
    parts, adv, label = [], [], ""
    for f, val in values.items():
        if not val:
            continue
        if f == "Condition":
            parts.append(f"query.cond={requests.utils.quote(str(val))}")
            label = str(val)
        elif f in ENUM_OPTIONS:
            vals = val if isinstance(val, list) else [val]
            sub = []
            for v in vals:
                if f == "HealthyVolunteers":
                    v = "true" if v == "Yes" else "false"
                sub.append(f"AREA[{f}]{v}")
            if sub:
                adv.append("(" + " OR ".join(sub) + ")")
        # MinimumAge / MaximumAge : post-filtre, pas dans l'URL
    if adv:
        parts.append("filter.advanced=" + requests.utils.quote(" AND ".join(adv)))
    if not parts:
        parts.append("query.cond=cancer")   # défaut de secours
    return "&".join(parts), (label or "recherche")


def parse_age_years(age_str):
    """'18 Years' -> 18 ; None si non parsable."""
    if not age_str:
        return None
    m = re.search(r"(\d+)", str(age_str))
    return int(m.group(1)) if m else None


def age_ok(study, user_min, user_max):
    """Garde l'étude si sa tranche d'âge chevauche [user_min, user_max]."""
    if not user_min and not user_max:
        return True
    elig = study.get("protocolSection", {}).get("eligibilityModule", {})
    s_min = parse_age_years(elig.get("minimumAge")) or 0
    s_max = parse_age_years(elig.get("maximumAge")) or 120
    if user_min and s_max < user_min:
        return False
    if user_max and s_min > user_max:
        return False
    return True


def build_rag_text(study):
    """Transforme le JSON ClinicalTrials en document texte structuré contenant
    TOUTES les sections utiles au RAG/NER (pas seulement l'éligibilité).
    Retourne "" si le JSON n'a aucun contenu exploitable -> fallback PDF."""
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    description = protocol.get("descriptionModule", {})
    conditions = protocol.get("conditionsModule", {})
    design = protocol.get("designModule", {})
    arms = protocol.get("armsInterventionsModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    outcomes = protocol.get("outcomesModule", {})

    sections = []

    def field(label, value):
        if value is None or value == "" or value == []:
            return ""
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        return f"{label}: {value}"

    def add_section(title, lines):
        clean = [str(l).strip() for l in lines if l is not None and str(l).strip()]
        if clean:
            sections.append(f"## {title}\n" + "\n".join(clean))

    add_section("Study identification", [
        field("NCT ID", identification.get("nctId")),
        field("Brief title", identification.get("briefTitle")),
        field("Official title", identification.get("officialTitle")),
    ])
    add_section("Study description", [
        field("Brief summary", description.get("briefSummary")),
        field("Detailed description", description.get("detailedDescription")),
    ])
    add_section("Conditions and keywords", [
        field("Conditions", conditions.get("conditions")),
        field("Keywords", conditions.get("keywords")),
    ])
    design_info = design.get("designInfo", {})
    add_section("Study design", [
        field("Study type", design.get("studyType")),
        field("Phases", design.get("phases")),
        field("Primary purpose", design_info.get("primaryPurpose")),
        field("Allocation", design_info.get("allocation")),
        field("Intervention model", design_info.get("interventionModel")),
        field("Observational model", design_info.get("observationalModel")),
    ])
    intervention_lines = []
    for idx, itv in enumerate(arms.get("interventions", []), start=1):
        intervention_lines.extend([
            field(f"Intervention {idx} type", itv.get("type")),
            field(f"Intervention {idx} name", itv.get("name")),
            field(f"Intervention {idx} description", itv.get("description")),
            field(f"Intervention {idx} arm labels", itv.get("armGroupLabels")),
        ])
    add_section("Interventions and treatments", intervention_lines)
    arm_lines = []
    for idx, arm in enumerate(arms.get("armGroups", []), start=1):
        arm_lines.extend([
            field(f"Arm {idx} label", arm.get("label")),
            field(f"Arm {idx} type", arm.get("type")),
            field(f"Arm {idx} description", arm.get("description")),
            field(f"Arm {idx} intervention names", arm.get("interventionNames")),
        ])
    add_section("Study arms", arm_lines)
    add_section("Eligibility", [
        field("Sex", eligibility.get("sex")),
        field("Minimum age", eligibility.get("minimumAge")),
        field("Maximum age", eligibility.get("maximumAge")),
        field("Healthy volunteers", eligibility.get("healthyVolunteers")),
        field("Eligibility criteria", eligibility.get("eligibilityCriteria")),
    ])
    outcome_lines = []
    for otype, olist in [("Primary outcome", outcomes.get("primaryOutcomes", [])),
                         ("Secondary outcome", outcomes.get("secondaryOutcomes", []))]:
        for idx, o in enumerate(olist, start=1):
            outcome_lines.extend([
                field(f"{otype} {idx} measure", o.get("measure")),
                field(f"{otype} {idx} description", o.get("description")),
                field(f"{otype} {idx} time frame", o.get("timeFrame")),
            ])
    add_section("Study outcomes", outcome_lines)

    # Garde-fou : au moins un contenu exploitable au-delà de l'identification
    has_useful_content = bool(
        description.get("briefSummary")
        or description.get("detailedDescription")
        or conditions.get("conditions")
        or arms.get("interventions")
        or eligibility.get("eligibilityCriteria")
        or outcomes.get("primaryOutcomes")
        or outcomes.get("secondaryOutcomes")
    )
    if not has_useful_content:
        return ""
    return "\n\n".join(sections)


def build_task(study, force_pdf):
    """Construit la tâche envoyée au backend. En mode 'text', on envoie le
    document complet reconstruit depuis le JSON (build_rag_text). Le PDF sert
    de fallback si l'utilisateur le force ou si le JSON n'a aucun contenu utile."""
    nct_id = _safe_get(study, "protocolSection", "identificationModule", "nctId")
    docs = _safe_get(study, "documentSection", "largeDocumentModule", "largeDocs") or []
    has_pdf = any(str(d.get("filename", "")).lower().endswith(".pdf") for d in docs)

    if force_pdf and has_pdf:
        return {"type": "pdf", "nct_id": nct_id}

    rag_text = build_rag_text(study)
    if rag_text:
        return {"type": "text", "nct_id": nct_id, "text": rag_text}

    return {"type": "pdf", "nct_id": nct_id}


def run_gpu_extraction(tasks, label, api_url, output_dir):
    """Envoie les études sélectionnées au serveur GPU (vLLM). Retourne les résultats."""
    results = []
    for task in tasks:
        nct_id = task["nct_id"]
        if nct_id in st.session_state.demo_cache:
            results.append(st.session_state.demo_cache[nct_id])
            continue
        try:
            if task["type"] == "text":
                resp = requests.post(
                    f"{api_url}/process_text",
                    data={"disease": label, "document_id": nct_id,
                          "text_content": task["text"]},
                    headers={"Bypass-Tunnel-Reminder": "true"})
            else:
                pdf_path = (download_pdf_for_nctid(nct_id, output_dir)
                            if download_pdf_for_nctid else None)
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as fh:
                        resp = requests.post(
                            f"{api_url}/process_pdf",
                            files={"file": (f"{nct_id}.pdf", fh, "application/pdf")},
                            data={"disease": label},
                            headers={"Bypass-Tunnel-Reminder": "true"})
                else:
                    st.warning(f"Ni texte ni PDF pour {nct_id}")
                    continue
            if resp.status_code == 200:
                data = resp.json()
                raw = data.get("extraction", "")
                if isinstance(raw, str):
                    m = re.search(r'(\{.*\})', raw.strip(), re.DOTALL)
                    try:
                        data["extraction"] = json.loads(m.group(1)) if m else {"raw": raw}
                    except Exception:
                        data["extraction"] = {"parse_error": "JSON invalide", "raw": raw}
                results.append(data)
                st.session_state.demo_cache[nct_id] = data
                st.session_state.extracted_docs.append(data.get("document", nct_id))
            else:
                st.error(f"Erreur API ({resp.status_code}) pour {nct_id}")
        except Exception as e:
            st.error(f"Impossible de traiter {nct_id} : {e}")
    return results


def results_to_df(results):
    """Aplatit les résultats d'extraction en tableau."""
    rows = []
    for res in results:
        ext = res.get("extraction", {})
        meds = ext.get("medications", []) if isinstance(ext, dict) else []
        meds = ", ".join(str(m.get("name", m.get("description", ""))) if isinstance(m, dict)
                         else str(m) for m in meds)
        crit = ext.get("inclusion_criteria", []) if isinstance(ext, dict) else []
        crit = ", ".join(str(c.get("description", c.get("category", ""))) if isinstance(c, dict)
                         else str(c) for c in crit)
        rows.append({
            "Document": res.get("document", "N/A"),
            "Pathologie": res.get("disease", "N/A"),
            "Condition Principale": ext.get("condition", "") if isinstance(ext, dict) else "",
            "Médicaments (Drug)": meds,
            "Critères (Measurement)": crit,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Onglets
# --------------------------------------------------------------------------- #
tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Critères de sélection",
    "📑 Summary table",
    "📊 Visualisation + filtre",
    "💬 Chatbot RAG",
])

# ============================ ONGLET 1 : RECHERCHE ========================= #
# Une seule requête ClinicalTrials. On stocke le JSON complet des études
# (found_studies) pour la Summary table -> pas de 2e requête. Aucune IA ici.
with tab1:
    st.header("1. Sélection des critères")
    st.markdown("Choisis **un ou plusieurs** critères, puis renseigne leurs valeurs.")

    label_to_field = {FIELD_LABELS.get(f, f): f for f in QUERY_FIELDS}
    sel_labels = st.multiselect("Critères de recherche :", list(label_to_field.keys()),
                                default=["Maladie (Condition)"])
    selected = [label_to_field[l] for l in sel_labels]

    values = {}
    cols = st.columns(2)
    for i, f in enumerate(selected):
        disp = FIELD_LABELS.get(f, f)
        with cols[i % 2]:
            if f in ENUM_OPTIONS:
                values[f] = st.multiselect(disp, ENUM_OPTIONS[f], key=f"val_{f}")
            elif f in ("MinimumAge", "MaximumAge"):
                values[f] = st.number_input(disp + " (années)", min_value=0, max_value=120,
                                            value=0, key=f"val_{f}")
            else:  # Condition (texte libre)
                values[f] = st.text_input(disp + " (ex: Breast Cancer)", key=f"val_{f}")

    c1, c2 = st.columns(2)
    max_results = c1.slider("Nombre d'essais à récupérer :", 1, 20, 5)
    force_pdf = c2.checkbox("📥 Forcer le scraping PDF (Plan B)")

    if st.button("🚀 Lancer la recherche", type="primary"):
        api_query, label = build_api_query(values)
        user_min = values.get("MinimumAge") or 0
        user_max = values.get("MaximumAge") or 0

        with st.spinner("Interrogation de l'API ClinicalTrials..."):
            t0 = time.time()
            kept = []
            url_ct = (f"https://clinicaltrials.gov/api/v2/studies?{api_query}"
                      f"&pageSize=50&fields=NCTId,ProtocolSection,DocumentSection")
            try:
                studies = requests.get(url_ct, timeout=15).json().get("studies", [])
                for study in studies:
                    if not age_ok(study, user_min, user_max):
                        continue
                    kept.append(study)
                    if len(kept) >= max_results:
                        break
            except Exception as e:
                st.error(f"Erreur API ClinicalTrials : {e}")
            elapsed = time.time() - t0

        if not kept:
            st.warning("Aucun essai trouvé pour ces critères.")
        else:
            # Nouvelle recherche -> on réinitialise l'analyse GPU précédente.
            st.session_state.found_studies = kept
            st.session_state.force_pdf = force_pdf
            st.session_state.latest_query = label
            st.session_state.search_done = True
            st.session_state.analysis_done = False
            st.session_state.latest_results = []
            st.session_state.extracted_docs = []
            st.session_state.selected_ncts = []
            st.success(f"✅ {len(kept)} essai(s) récupéré(s) en {elapsed:.1f}s. "
                       "Va dans l'onglet « 📑 Summary table ».")


# ============================ ONGLET 2 : SUMMARY TABLE ===================== #
# 100% ClinicalTrials, aucun GPU. C'est ici qu'on coche les études à analyser,
# puis qu'on déclenche l'extraction GPU (qui débloque les onglets 3 & 4).
with tab2:
    st.header("2. Summary table (champs source ClinicalTrials)")
    if not st.session_state.search_done:
        lock_need_search()
    else:
        rows = [study_summary_row(s) for s in st.session_state.found_studies]
        df_sum = pd.DataFrame(rows)
        # Pré-coche les études déjà sélectionnées (restaurées au refresh)
        _prev = set(st.session_state.get("selected_ncts", []))
        df_sum.insert(0, "Analyser", df_sum["NCT_ID"].isin(_prev))

        st.caption("Coche les études à envoyer au modèle, puis clique « Analyser (GPU) ».")
        edited = st.data_editor(
            df_sum, hide_index=True, use_container_width=True,
            column_config={"Analyser": st.column_config.CheckboxColumn(
                "Analyser", help="Cocher pour envoyer cette étude au GPU")},
            disabled=[c for c in df_sum.columns if c != "Analyser"],
            key="summary_editor")

        selected_ncts = edited.loc[edited["Analyser"] == True, "NCT_ID"].tolist()
        st.session_state.selected_ncts = selected_ncts

        st.download_button("📥 Exporter la table (CSV)",
                           df_sum.drop(columns=["Analyser"]).to_csv(index=False, sep=';').encode('utf-8-sig'),
                           file_name="summary_table.csv", mime="text/csv")

        st.markdown("---")
        cbtn, cinfo = st.columns([1, 2])
        launch = cbtn.button(f"🚀 Analyser {len(selected_ncts)} étude(s) cochée(s) (GPU)",
                             type="primary", disabled=not selected_ncts)
        cinfo.caption("Nécessite le serveur GPU allumé. Débloque les onglets 3 & 4.")

        if launch:
            label = st.session_state.latest_query
            output_dir = os.path.abspath(f"data/live_pdfs_{label.replace(' ', '_')}")
            os.makedirs(output_dir, exist_ok=True)
            sel_studies = [s for s in st.session_state.found_studies
                           if _safe_get(s, "protocolSection", "identificationModule", "nctId") in selected_ncts]
            tasks = [build_task(s, st.session_state.force_pdf) for s in sel_studies]
            with st.spinner(f"Envoi de {len(tasks)} protocole(s) au GPU (vLLM)..."):
                t0 = time.time()
                results = run_gpu_extraction(tasks, label, api_url, output_dir)
                st.session_state.latest_results = results
                st.session_state.extracted_docs = list(set(st.session_state.extracted_docs))
                if results:
                    st.session_state.analysis_done = True
                    st.success(f"✅ Extraction terminée en {time.time() - t0:.1f}s — "
                               "onglets 3 & 4 débloqués.")
                else:
                    st.warning("Aucun résultat renvoyé par le GPU (serveur éteint ?).")

        if st.session_state.analysis_done and st.session_state.latest_results:
            with st.expander("🔎 Aperçu des extractions générées"):
                for res in st.session_state.latest_results:
                    st.markdown(f"**{res.get('document', 'N/A')}** — {res.get('disease', 'N/A')}")
                    st.json(res.get("extraction", {}))


# ============================ ONGLET 3 : VISUALISATION ===================== #
with tab3:
    st.header("3. Visualisation & filtre des résultats")
    if not st.session_state.analysis_done:
        lock_need_analysis()
    else:
        df = results_to_df(st.session_state.latest_results)

        # --- Filtre post-JSON (champs de Jérémie) ---
        with st.expander("🔎 Filtrer les résultats (filtre post-JSON)", expanded=True):
            st.caption("Champs de filtre : " + ", ".join(FILTER_FIELDS))
            txt = st.text_input("Filtre texte (cherche dans Condition / Médicaments / Critères) :")
        if txt:
            mask = df.apply(lambda row: txt.lower() in " ".join(row.astype(str)).lower(), axis=1)
            df = df[mask]

        st.subheader("📊 Tableau de bord clinique")
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Exporter les extractions (CSV)",
                           df.to_csv(index=False, sep=';').encode('utf-8-sig'),
                           file_name="extractions.csv", mime="text/csv")

        # --- Petit graphique : nb d'entités par type ---
        counts = {"Condition": 0, "Médicaments": 0, "Critères": 0}
        for _, row in df.iterrows():
            counts["Condition"] += 1 if row["Condition Principale"] else 0
            counts["Médicaments"] += len([m for m in str(row["Médicaments (Drug)"]).split(",") if m.strip()])
            counts["Critères"] += len([c for c in str(row["Critères (Measurement)"]).split(",") if c.strip()])
        st.subheader("Répartition des entités extraites")
        st.bar_chart(pd.DataFrame({"Nombre": counts}))


# ============================ ONGLET 4 : RAG =============================== #
with tab4:
    st.header("4. Assistant Chatbot RAG")
    if not st.session_state.analysis_done:
        lock_need_analysis()
    else:
        doc_filter = st.selectbox("Filtrer par essai (Optionnel) :",
                                  ["Toute la base"] + st.session_state.extracted_docs)
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("Posez votre question clinique..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                ph = st.empty()
                ph.markdown("🧠 Recherche vectorielle + génération...")
                try:
                    sel = None if doc_filter == "Toute la base" else doc_filter
                    r = requests.post(f"{api_url}/chat_rag",
                                      data={"question": prompt, "doc_id": sel},
                                      headers={"Bypass-Tunnel-Reminder": "true"})
                    if r.status_code == 200:
                        ans = r.json().get("answer", "Erreur de génération.")
                        ph.markdown(ans)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})
                    else:
                        ph.error(f"Erreur API ({r.status_code})")
                except Exception as e:
                    ph.error(f"Impossible de joindre l'API : {e}")
