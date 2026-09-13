import numpy as np
from typing import Dict, Any, Union
from deepface import DeepFace
from src.config import MODEL_NAME, DETECTOR_BACKEND, DEFAULT_THRESHOLD
from src.database import load_database

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1D vectors.
    Returns a value between -1 and 1. Higher is more similar.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return float(dot_product / (norm_vec1 * norm_vec2))

def identify_face(image_path_or_array: Union[str, Any], threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """
    Identifies a face in the given image.
    Returns a dictionary with status, identity, similarity, and threshold.
    """
    
    db = load_database()
    if not db:
        return {
            "status": "error",
            "message": "Database is empty. Please enroll someone first.",
            "identity": None,
            "similarity": None,
            "threshold": threshold
        }
    
    # 1. Extract Embedding
    try:
        results = DeepFace.represent(
            img_path=image_path_or_array,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
        
        if isinstance(results, dict):
            results = [results]
            
        num_faces = len(results)
        if num_faces == 0:
            return {"status": "error", "message": "No face detected in the image."}
        elif num_faces > 1:
            return {"status": "error", "message": f"Multiple faces ({num_faces}) detected. Exactly one required."}
            
        query_embedding = np.array(results[0]["embedding"])
        
    except ValueError as e:
        if "Face could not be detected" in str(e):
            return {"status": "error", "message": "No face detected in the image."}
        return {"status": "error", "message": f"Validation Error: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}
        
    # 2. Compare against all enrolled identities
    best_overall_similarity = -1.0
    best_candidate = None
    
    for person_name, data in db.items():
        embeddings = data.get("embeddings", [])
        if not embeddings:
            continue
            
        # Calculate similarity against each stored embedding for this person
        person_best_similarity = -1.0
        for stored_emb in embeddings:
            sim = compute_cosine_similarity(query_embedding, np.array(stored_emb))
            if sim > person_best_similarity:
                person_best_similarity = sim
                
        # Compare person's best score with overall best score
        if person_best_similarity > best_overall_similarity:
            best_overall_similarity = person_best_similarity
            best_candidate = person_name
            
    # 3. Apply threshold logic
    if best_candidate is not None and best_overall_similarity >= threshold:
        return {
            "status": "matched",
            "identity": best_candidate,
            "similarity": best_overall_similarity,
            "threshold": threshold
        }
    else:
        return {
            "status": "unknown",
            "identity": None,
            "similarity": best_overall_similarity if best_candidate is not None else 0.0,
            "threshold": threshold
        }
