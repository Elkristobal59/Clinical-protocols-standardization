import argparse
import json
import os
from collections import defaultdict

try:
    import mlflow
except ImportError:
    mlflow = None

def compute_metrics(gold_data, pred_data, match_type="text"):
    """
    Calcule la Précision, le Recall et le F1-Score pour chaque type d'entité.
    """
    metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    
    # Construction de dictionnaires pour un accès rapide
    # Format attendu : { "doc_id": [ {"type": "Condition", "text": "Breast Cancer"}, ... ] }
    gold_dict = {doc.get("id", i): doc.get("entities", []) for i, doc in enumerate(gold_data)}
    pred_dict = {doc.get("id", i): doc.get("entities", []) for i, doc in enumerate(pred_data)}
    
    for doc_id, gold_entities in gold_dict.items():
        pred_entities = pred_dict.get(doc_id, [])
        
        # Copie pour le matching
        unmatched_gold = list(gold_entities)
        unmatched_pred = list(pred_entities)
        
        # Support
        for ge in gold_entities:
            metrics[ge["type"]]["support"] += 1
            
        # Vrais Positifs (TP)
        for pe in list(unmatched_pred):
            match_found = False
            for ge in list(unmatched_gold):
                if pe["type"] == ge["type"]:
                    # Exact Match ou Relaxed Match
                    if match_type == "text" and pe.get("text", "").lower().strip() == ge.get("text", "").lower().strip():
                        match_found = True
                    elif match_type == "relaxed" and (pe.get("text", "").lower().strip() in ge.get("text", "").lower().strip() or ge.get("text", "").lower().strip() in pe.get("text", "").lower().strip()):
                        match_found = True
                        
                if match_found:
                    metrics[pe["type"]]["tp"] += 1
                    unmatched_gold.remove(ge)
                    unmatched_pred.remove(pe)
                    break
                    
        # Faux Positifs (FP) - Les entités prédites qui n'ont pas de match
        for pe in unmatched_pred:
            metrics[pe["type"]]["fp"] += 1
            
        # Faux Négatifs (FN) - Les entités gold qui n'ont pas été trouvées
        for ge in unmatched_gold:
            metrics[ge["type"]]["fn"] += 1

    # Calcul des pourcentages
    results = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for entity_type, counts in metrics.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        support = counts["support"]
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[entity_type] = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support
        }
        
    # Micro Average
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    
    results["MICRO"] = {
        "Precision": micro_p,
        "Recall": micro_r,
        "F1": micro_f1,
        "Support": sum(c["support"] for c in metrics.values())
    }
    
    return results

def print_results(results, system_name, match_type):
    print(f"\n=== Benchmark NER - système : {system_name}  (matching = {match_type}) ===")
    print(f"{'Type':<15} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
    print("-" * 45)
    
    for entity_type, metrics in results.items():
        if entity_type == "MICRO":
            continue
        print(f"{entity_type:<15} {metrics['Precision']:.3f} {metrics['Recall']:.3f} {metrics['F1']:.3f} {metrics['Support']:8}")
        
    print("-" * 45)
    micro = results.get("MICRO", {})
    print(f"{'MICRO':<15} {micro.get('Precision', 0):.3f} {micro.get('Recall', 0):.3f} {micro.get('F1', 0):.3f}")
    
    # Macro F1 (Average of all F1s except MICRO)
    types_only = [m["F1"] for k, m in results.items() if k != "MICRO"]
    macro_f1 = sum(types_only) / len(types_only) if types_only else 0.0
    print(f"{'MACRO-F1':<15} {'':>6} {'':>6} {macro_f1:.3f}")
    print("\n[ok] Résultats détaillés -> benchmark_results.json")

def main():
    parser = argparse.ArgumentParser(description="Script d'évaluation NER (Precision/Recall/F1)")
    parser.add_argument("--gold", type=str, required=True, help="Chemin vers le fichier JSON de référence (Gold Standard)")
    parser.add_argument("--pred", type=str, required=True, help="Chemin vers le fichier JSON des prédictions du LLM")
    parser.add_argument("--system", type=str, default="Qwen", help="Nom du système (ex: Qwen)")
    parser.add_argument("--match", type=str, choices=["text", "relaxed"], default="text", help="Méthode de matching (exact ou relaxed)")
    parser.add_argument("--types-file", type=str, help="Fichier txt avec les types autorisés (optionnel)")
    parser.add_argument("--mlflow", action="store_true", help="Activer le log vers MLflow")
    
    args = parser.parse_args()
    
    # Chargement des données
    try:
        with open(args.gold, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
        with open(args.pred, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement des fichiers JSON : {e}")
        print("Assurez-vous que les fichiers ont la structure suivante :")
        print('[{"id": "doc1", "entities": [{"type": "Condition", "text": "Breast Cancer"}]}]')
        return

    # Calcul des métriques
    results = compute_metrics(gold_data, pred_data, match_type=args.match)
    
    # Affichage Console
    print_results(results, args.system, args.match)
    
    # Sauvegarde JSON
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    # Log MLflow
    if args.mlflow:
        if mlflow is None:
            print("[erreur] MLflow n'est pas installé. (pip install mlflow)")
        else:
            mlflow.set_experiment("NER_Benchmark")
            with mlflow.start_run(run_name=f"Eval_{args.system}_{args.match}"):
                mlflow.log_param("system", args.system)
                mlflow.log_param("match_type", args.match)
                
                # Log metrics for each type
                for entity_type, metrics in results.items():
                    clean_type = entity_type.replace(" ", "_").replace("-", "_")
                    mlflow.log_metric(f"{clean_type}_Precision", metrics["Precision"])
                    mlflow.log_metric(f"{clean_type}_Recall", metrics["Recall"])
                    mlflow.log_metric(f"{clean_type}_F1", metrics["F1"])
                    
                print("[ok] Métriques envoyées à MLflow.")

if __name__ == "__main__":
    main()
