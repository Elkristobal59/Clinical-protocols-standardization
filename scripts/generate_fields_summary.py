"""
Script : generate_fields_summary.py (Le Formateur Streamlit)
-----------------------------------------------------------
Rôle : Interroger l'API officielle ClinicalTrials.gov (v2) et extraire uniquement 
les métadonnées clés pour les afficher proprement dans l'interface utilisateur (Streamlit).

🎓 Explication pour le jury :
Les réponses de l'API de ClinicalTrials sont des JSON géants (parfois 3000 lignes pour un seul essai !).
L'application Streamlit n'a pas besoin de toute cette complexité.
Ce script agit comme un "entonnoir" (Data Transformation). Il prend le JSON géant, navigue dans 
l'arborescence complexe, et ressort un dictionnaire ultra-propre avec juste les 10 colonnes demandées 
par le métier (Titre, Phase, Maladie, etc.).
"""

import requests
import pandas as pd
import argparse
import sys
import os

def safe_get(data, *keys):
    """
    Fonction utilitaire pour naviguer dans le JSON de l'API v2 en toute sécurité.
    Évite les erreurs fatales (KeyError) si une étude n'a pas renseigné un champ spécifique.
    """
    curr = data
    for k in keys:
        if isinstance(curr, dict) and k in curr:
            curr = curr[k]
        else:
            return None
    return curr

def has_pdf(nct_id):
    """
    Vérifie si un PDF est présent pour ce NCT ID dans notre base de données locale (data/chia_pdfs).
    Permet d'afficher une jolie icône "✅" dans Streamlit si le document est prêt pour l'IA.
    """
    pdf_dir = os.path.join("data", "chia_pdfs")
    if os.path.exists(pdf_dir):
        for filename in os.listdir(pdf_dir):
            if filename.startswith(nct_id) and filename.endswith(".pdf"):
                return "Oui"
    return "Non"

def extract_study_data(study_json):
    """
    Extrait exactement les champs demandés (Business Requirements) depuis la structure 
    complexe de l'API v2.
    """
    
    # --- Champs d'Identification ---
    nct_id = safe_get(study_json, "protocolSection", "identificationModule", "nctId")
    title = safe_get(study_json, "protocolSection", "identificationModule", "officialTitle")
    
    # --- Design & Type ---
    study_type = safe_get(study_json, "protocolSection", "designModule", "studyType")
    phases = safe_get(study_json, "protocolSection", "designModule", "phases")
    phase_str = ", ".join(phases) if phases else "N/A"
    
    primary_purpose = safe_get(study_json, "protocolSection", "designModule", "designInfo", "primaryPurpose")
    enrollment = safe_get(study_json, "protocolSection", "designModule", "enrollmentInfo", "count")
    
    # --- Critères d'éligibilité ---
    eligibility = safe_get(study_json, "protocolSection", "eligibilityModule", "eligibilityCriteria")
    eligibility_status = "Présent" if eligibility else "Absent"
    
    # --- Interventions (Il peut y en avoir plusieurs) ---
    interventions = safe_get(study_json, "protocolSection", "armsInterventionsModule", "interventions")
    int_names = []
    int_types = []
    if interventions:
        for inter in interventions:
            name = inter.get("name")
            itype = inter.get("type")
            if name: int_names.append(name)
            if itype: int_types.append(itype)
            
    int_name_str = " | ".join(int_names) if int_names else "N/A"
    int_type_str = " | ".join(int_types) if int_types else "N/A"
    
    # --- Primary Outcomes (Ce qu'on mesure) ---
    outcomes = safe_get(study_json, "protocolSection", "outcomesModule", "primaryOutcomes")
    outcome_measures = []
    if outcomes:
        for out in outcomes:
            measure = out.get("measure")
            if measure: outcome_measures.append(measure)
    outcome_str = " | ".join(outcome_measures) if outcome_measures else "N/A"
    
    # 📝 Retourne la ligne structurée prête pour le DataFrame Pandas
    return {
        "NCT_ID": nct_id,
        "PDF Présent": has_pdf(nct_id),
        "Official Title": title or "N/A",
        "Study Type": study_type or "N/A",
        "Phase": phase_str,
        "Primary Purpose": primary_purpose or "N/A",
        "Enrollment Count": enrollment or "N/A",
        "Eligibility Criteria": eligibility_status,
        "Intervention Type": int_type_str,
        "Intervention Name": int_name_str,
        "Primary Outcome Measure": outcome_str
    }

def main():
    parser = argparse.ArgumentParser(description="Générer une table de résumé des essais cliniques")
    parser.add_argument("--query", type=str, help="Mot clé de la pathologie (ex: 'Breast Cancer')")
    parser.add_argument("--nct_ids", type=str, help="Liste d'IDs séparés par des virgules")
    parser.add_argument("--output", type=str, default="summary_table_jeremie.csv", help="Fichier de sortie CSV")
    
    args = parser.parse_args()
    
    if not args.query and not args.nct_ids:
        print("Erreur : Veuillez spécifier --query ou --nct_ids")
        sys.exit(1)
        
    studies_data = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    if args.query:
        print(f"Recherche des études pour la requête : {args.query}")
        url = f"https://clinicaltrials.gov/api/v2/studies?query.cond={args.query}&pageSize=10"
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                raw_studies = resp.json().get("studies", [])
                for s in raw_studies:
                    studies_data.append(extract_study_data(s))
        except Exception as e:
            print(f"Erreur API: {e}")
            sys.exit(1)
            
    elif args.nct_ids:
        ids = [x.strip() for x in args.nct_ids.split(",")]
        for nct_id in ids:
            print(f"Récupération de {nct_id}...")
            url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    studies_data.append(extract_study_data(resp.json()))
            except Exception as e:
                print(f"Erreur API pour {nct_id}: {e}")
                
    if not studies_data:
        print("Aucune étude trouvée.")
        sys.exit(0)
        
    print(f"{len(studies_data)} études récupérées. Génération du tableau...")
    # 🐼 Conversion en Dataframe pour un affichage propre ou export CSV
    df = pd.DataFrame(studies_data)
    df.to_csv(args.output, index=False, sep=";", encoding="utf-8-sig")
    print(f"✅ Tableau généré avec succès dans : {args.output}")

if __name__ == "__main__":
    main()
