import os
import torch

class Config:
    # Project Root Directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploads")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "run_active")
    LITERATURE_DIR = os.path.join(BASE_DIR, "data", "literature")

    # PyTorch Hardware Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_BATCH_SIZE = 32  # DL Inference Batch Size

    # File Format Restrictions (Standard PDB & SDF for DL)
    ALLOWED_PROTEIN_EXTENSIONS = {'pdb'}
    ALLOWED_LIGAND_EXTENSIONS = {'sdf', 'mol2'}

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    # RAG Settings
    TOP_N_FOR_RAG = 5
