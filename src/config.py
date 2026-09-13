import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "database.json")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")

# Ensure data and eval directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# DeepFace Models Config
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
EMBEDDING_DIM = 512

# Cosine Similarity Threshold (configurable). 
# Calibrated to 0.45 based on the actual demo dataset to balance 
# max impostor score (0.392) and min genuine max-score (0.495).
DEFAULT_THRESHOLD = 0.45
