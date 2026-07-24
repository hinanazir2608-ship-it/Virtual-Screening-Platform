import os
import logging

logger = logging.getLogger("MDComplexGenerator")

def create_complex_pdb(receptor_pdb: str, ligand_pdb: str, output_complex_pdb: str) -> bool:
    """
    Combines clean receptor PDB and PyTorch docked ligand PDB into a single combined PDB complex.
    """
    try:
        lines = []
        
        # 1. Parse Receptor PDB ATOMs
        if os.path.exists(receptor_pdb):
            with open(receptor_pdb, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(("ATOM", "HETATM")):
                        lines.append(line)
        
        lines.append("TER\n")

        # 2. Parse Docked Ligand PDB ATOMs
        if os.path.exists(ligand_pdb):
            with open(ligand_pdb, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(("ATOM", "HETATM")):
                        lines.append(line)

        lines.append("END\n")

        os.makedirs(os.path.dirname(output_complex_pdb), exist_ok=True)
        with open(output_complex_pdb, "w", encoding="utf-8") as f:
            f.writelines(lines)
            
        return True
    except Exception as e:
        logger.error(f"Error merging complex: {e}")
        return False
