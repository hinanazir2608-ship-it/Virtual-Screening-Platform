import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger("ResultsEngine")

def scan_and_rank_results(docking_out_dir: str, summary_json_path: str = None) -> List[Dict[str, Any]]:
    """
    PyTorch Docking Results Scanner:
    Reads predicted affinity scores directly from JSON summary or log outputs.
    """
    results = []
    
    # 1. Check if PyTorch engine saved a summary JSON
    if summary_json_path and os.path.exists(summary_json_path):
        try:
            with open(summary_json_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            # Sort by affinity (assuming lower/more negative is better kcal/mol)
            results.sort(key=lambda x: x.get("affinity", 0.0))
            return results
        except Exception as e:
            logger.error(f"Error reading summary json: {e}")

    # 2. Fallback: Scan docking output directory for PyTorch output logs/PDBs
    if os.path.exists(docking_out_dir):
        for file in os.listdir(docking_out_dir):
            if file.endswith("_log.txt"):
                ligand_id = file.replace("_log.txt", "")
                log_path = os.path.join(docking_out_dir, file)
                out_pdb = os.path.join(docking_out_dir, f"{ligand_id}_docked.pdb")
                
                affinity = 0.0
                try:
                    with open(log_path, "r") as f:
                        for line in f:
                            if "Affinity:" in line:
                                affinity = float(line.split(":")[1].replace("kcal/mol", "").strip())
                except Exception:
                    pass

                results.append({
                    "compound_id": ligand_id,
                    "affinity": affinity,
                    "out_pdb": out_pdb,
                    "log_file": log_path
                })

    # Sort results (best affinity first)
    results.sort(key=lambda x: x.get("affinity", 0.0))
    return results

def write_results_json(results: List[Dict[str, Any]], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

def write_results_csv(results: List[Dict[str, Any]], output_path: str):
    import csv
    if not results:
        return
    keys = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
