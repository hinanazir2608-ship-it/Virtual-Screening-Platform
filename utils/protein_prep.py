import os
import re
import subprocess
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger("ProteinPrep")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ProteinPreparationError(Exception):
    pass

def download_pdb(pdb_id: str, output_dir: str) -> str:
    pdb_id = pdb_id.strip().lower()
    if len(pdb_id) != 4:
        raise ProteinPreparationError(f"Invalid PDB ID format: '{pdb_id}'")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        import urllib.request
        urllib.request.urlretrieve(url, output_path)
        return output_path
    except Exception as e:
        raise ProteinPreparationError(f"Failed to download PDB ID '{pdb_id}': {str(e)}")

def clean_pdb_structure(input_pdb: str, output_pdb: str, keep_water: bool = False, keep_hetero: bool = False) -> str:
    if not os.path.exists(input_pdb):
        raise ProteinPreparationError(f"Input PDB file not found: {input_pdb}")

    cleaned_lines = []
    with open(input_pdb, "r", encoding="utf-8") as infile:
        for line in infile:
            record_type = line[:6].strip()
            if record_type == "ATOM":
                cleaned_lines.append(line)
            elif record_type == "HETATM":
                res_name = line[17:20].strip()
                if keep_water and res_name in ["HOH", "WAT"]:
                    cleaned_lines.append(line)
                elif keep_hetero and res_name not in ["HOH", "WAT"]:
                    cleaned_lines.append(line)
            elif record_type in ["TER", "END"]:
                cleaned_lines.append(line)

    if not cleaned_lines:
        raise ProteinPreparationError("No valid atomic structural data remained after cleaning.")

    os.makedirs(os.path.dirname(output_pdb), exist_ok=True)
    with open(output_pdb, "w", encoding="utf-8") as outfile:
        outfile.writelines(cleaned_lines)

    return output_pdb

def prepare_receptor(
    receptor_source: str,
    output_dir: str,
    ph: float = 7.4,
    keep_water: bool = False,
    keep_hetero: bool = False
) -> Dict[str, Any]:
    """
    PyTorch Pipelines ke liye Optimized Pipeline:
    Ye direct clean `.pdb` structure return karega (PDBQT ki bajaye).
    """
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(receptor_source):
        raw_pdb_path = receptor_source
        base_name = os.path.splitext(os.path.basename(receptor_source))[0]
    else:
        raw_pdb_path = download_pdb(receptor_source, output_dir)
        base_name = receptor_source.lower()

    cleaned_pdb = os.path.join(output_dir, f"{base_name}_clean.pdb")

    # Clean Structure (PDBQT conversion completely bypassed)
    clean_pdb_structure(raw_pdb_path, cleaned_pdb, keep_water=keep_water, keep_hetero=keep_hetero)

    logger.info(f"Receptor preparation complete (PDB format): {cleaned_pdb}")
    
    return {
        "status": "success",
        "receptor_id": base_name,
        "clean_pdb": cleaned_pdb
    }
