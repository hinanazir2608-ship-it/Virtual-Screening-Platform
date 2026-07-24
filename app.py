import os
import json
import threading
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

# 1. Project root se .env file ko load karein
load_dotenv()

# 2. Key Status Check & Print
groq_key = os.getenv("GROQ_API_KEY")
print("-" * 50)
if groq_key:
    print("GROQ KEY STATUS: LOADED! ✅")
else:
    print("GROQ KEY STATUS: NOT LOADED! ❌")

# 3. PyTorch Detection & GPU Status Check
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU"
    print(f"PYTORCH STATUS: LOADED! ✅ (CUDA/GPU: {CUDA_AVAILABLE} - Device: {device_name})")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    print("PYTORCH STATUS: NOT INSTALLED! ❌")
print("-" * 50)

# Existing Core Pipeline Modules
from utils.protein_prep import prepare_receptor
from utils.ligand_prep import process_ligand_batch
from utils.docking_engine import run_vina_docking
from utils.report_orchestrator import rank_results, finalize_report

# RAG & Literature Imports
from utils.hit_literature_fetcher import fetch_hit_literature

# Groq AI Assistant Import
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

app = Flask(__name__)

# Folders Setup
UPLOAD_FOLDER = os.path.join("data", "uploads")
OUTPUT_FOLDER = os.path.join("outputs", "run_active")

PROTEIN_DIR = os.path.join(UPLOAD_FOLDER, "protein")
LIGAND_DIR = os.path.join(UPLOAD_FOLDER, "ligand")
CONFIG_DIR = os.path.join(UPLOAD_FOLDER, "config")

os.makedirs(PROTEIN_DIR, exist_ok=True)
os.makedirs(LIGAND_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Global Progress Control
PROGRESS_LOCK = threading.Lock()
PROGRESS = {
    "status": "idle",
    "progress": 0,
    "current_compound": "",
    "stage": 0,
    "total_compounds": 0
}

def update_progress(stage, progress, compound="", status="running"):
    with PROGRESS_LOCK:
        PROGRESS["stage"] = stage
        PROGRESS["progress"] = progress
        PROGRESS["current_compound"] = compound
        PROGRESS["status"] = status

def parse_vina_config(config_file_path):
    """
    Vina conf.txt file ko parse karta hai aur center_x/y/z 
    aur size_x/y/z coordinates extract karta hai.
    """
    grid_center = [0.0, 0.0, 0.0]
    box_size = [20.0, 20.0, 20.0] # Default fallback size
    
    if not config_file_path or not os.path.exists(config_file_path):
        print(f"[WARNING] Config file missing/not found at: {config_file_path}. Using defaults.")
        return tuple(grid_center), tuple(box_size)

    try:
        with open(config_file_path, 'r') as f:
            lines = f.readlines()
            
        config_dict = {}
        for line in lines:
            line = line.split('#')[0].strip()
            if '=' in line:
                key, val = line.split('=', 1)
                config_dict[key.strip().lower()] = float(val.strip())

        if 'center_x' in config_dict and 'center_y' in config_dict and 'center_z' in config_dict:
            grid_center = [
                config_dict['center_x'],
                config_dict['center_y'],
                config_dict['center_z']
            ]
            
        if 'size_x' in config_dict and 'size_y' in config_dict and 'size_z' in config_dict:
            box_size = [
                config_dict['size_x'],
                config_dict['size_y'],
                config_dict['size_z']
            ]

        print(f"[SUCCESS] Parsed Vina Config -> Center: {grid_center}, Size: {box_size}")
    except Exception as e:
        print(f"[ERROR] Failed to parse config file: {str(e)}")

    return tuple(grid_center), tuple(box_size)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/progress")
def get_progress():
    with PROGRESS_LOCK:
        return jsonify(PROGRESS)

@app.route("/api/results")
def get_results():
    results_file = os.path.join(OUTPUT_FOLDER, "results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return jsonify(json.load(f))
    return jsonify([])

# System / PyTorch Status Endpoint
@app.route("/api/system-status")
def system_status():
    return jsonify({
        "pytorch_installed": TORCH_AVAILABLE,
        "cuda_available": CUDA_AVAILABLE,
        "device": torch.cuda.get_device_name(0) if CUDA_AVAILABLE else ("CPU" if TORCH_AVAILABLE else "None"),
        "groq_available": GROQ_AVAILABLE
    })

def handle_upload(request_obj, target_dir):
    try:
        if "file" not in request_obj.files:
            return jsonify({"status": "error", "message": "No file field in request"}), 400
        
        file = request_obj.files["file"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(target_dir, filename)
        file.save(filepath)
        
        return jsonify({
            "status": "success", 
            "filename": filename, 
            "filepath": filepath
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/upload/protein", methods=["POST"])
def upload_protein():
    return handle_upload(request, PROTEIN_DIR)

@app.route("/api/upload/ligand", methods=["POST"])
def upload_ligand():
    return handle_upload(request, LIGAND_DIR)

@app.route("/api/upload/config", methods=["POST"])
def upload_config():
    return handle_upload(request, CONFIG_DIR)

@app.route("/api/upload", methods=["POST"])
def upload_generic():
    return handle_upload(request, UPLOAD_FOLDER)

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "message": "API Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"status": "error", "message": "Internal server error occurred"}), 500

def run_pipeline_thread(config):
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        update_progress(1, 10, "Preparing Target Protein")
        receptor_file = config.get("receptor_path") or config.get("receptor_id")
        rec_info = prepare_receptor(receptor_file, OUTPUT_FOLDER, ph=7.4)

        update_progress(2, 30, "Preparing Ligand Library Batch")
        sdf_file = config.get("sdf_path")
        ligands = process_ligand_batch(sdf_file, os.path.join(OUTPUT_FOLDER, "ligands"))

        total = len(ligands)
        docking_dir = os.path.join(OUTPUT_FOLDER, "docking_out")
        os.makedirs(docking_dir, exist_ok=True)

        for idx, lig in enumerate(ligands):
            prog = 30 + int((idx / max(total, 1)) * 45)
            update_progress(3, prog, f"Docking compound {idx+1}/{total}")
            run_vina_docking(
                rec_info["receptor_pdbqt"], 
                lig, 
                docking_dir, 
                grid_center=config.get("grid_center", (0.0, 0.0, 0.0))
            )

        update_progress(5, 80, "Ranking Docking Results")
        results = rank_results(OUTPUT_FOLDER)

        top_hits = [hit.get("compound_id", "CMNPD23787") for hit in results[:5]] if results else ["CMNPD23787"]

        update_progress(5, 86, "Fetching Literature for Top Hits")
        for comp_id in top_hits:
            fetch_hit_literature(comp_id)

        update_progress(5, 92, "Running RAG Analysis & Building Report")
        finalize_report(OUTPUT_FOLDER, corpus_dir="data/literature", metadata=config, results=results)

        update_progress(5, 100, "Screening complete - results and top complexes ready.", status="completed")

    except Exception as e:
        update_progress(0, 0, f"Error: {str(e)}", status="failed")

@app.route('/api/start', methods=['POST'])
def start_screening():
    data = request.get_json() or {}
    
    receptor_path = data.get('receptor_path', '')
    sdf_path = data.get('sdf_path', '')
    config_path = data.get('config_path', '')

    grid_center, box_size = parse_vina_config(config_path)

    pipeline_config = {
        'receptor_path': receptor_path,
        'sdf_path': sdf_path,
        'config_path': config_path,
        'grid_center': grid_center,
        'box_size': box_size
    }

    return jsonify({
        "status": "success",
        "message": "Screening started with parsed grid parameters.",
        "grid_center": grid_center,
        "box_size": box_size
    })

@app.route("/api/chat", methods=["POST"])
def ai_chat():
    data = request.json or {}
    user_prompt = data.get("prompt", "")
    
    if not GROQ_AVAILABLE:
        return jsonify({"response": "Groq package missing. Please install via pip install groq"}), 500
        
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return jsonify({"response": "GROQ_API_KEY environment variable set nahi hai."}), 400

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert AI Research Assistant specialized in computational biochemistry and virtual screening analysis."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        answer = completion.choices[0].message.content
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"response": f"Error communicating with AI Assistant: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
