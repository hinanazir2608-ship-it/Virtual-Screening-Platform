import os
import gc
import logging
from typing import Generator, List

logger = logging.getLogger("LigandPrep")

def parse_sdf_molecules(sdf_path: str) -> Generator[str, None, None]:
    """Multi-compound SDF file ko split karne ke liye generator."""
    current_mol = []
    with open(sdf_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            current_mol.append(line)
            if line.startswith("$$$$"):
                yield "".join(current_mol)
                current_mol = []
    if current_mol:
        yield "".join(current_mol)

def save_sdf_block(sdf_block: str, mol_id: str, output_dir: str) -> str:
    """
    PDBQT conversion bypass: Direct single compound SDF save karega.
    PyTorch models Native SDF files fast process karte hain.
    """
    output_sdf = os.path.join(output_dir, f"{mol_id}.sdf")
    try:
        with open(output_sdf, "w", encoding="utf-8") as f:
            f.write(sdf_block)
        return output_sdf
    except Exception as e:
        logger.error(f"Failed to save ligand {mol_id}: {e}")
        return None

def process_ligand_batch(sdf_file: str, output_dir: str, batch_size: int = 500) -> List[str]:
    """Individual `.sdf` files ka batch tayar karta hai."""
    os.makedirs(output_dir, exist_ok=True)
    prepared_files = []
    count = 0

    for idx, mol_block in enumerate(parse_sdf_molecules(sdf_file)):
        mol_id = f"LIG_{idx+1:05d}"
        sdf_filepath = save_sdf_block(mol_block, mol_id, output_dir)
        if sdf_filepath:
            prepared_files.append(sdf_filepath)
        
        count += 1
        if count % batch_size == 0:
            gc.collect()

    gc.collect()
    return prepared_files
