import os
from typing import Dict, Any, Union
from deepface import DeepFace
from src.config import MODEL_NAME, DETECTOR_BACKEND
from src.database import add_embedding_to_person

def process_and_enroll_image(person_name: str, image_path_or_array: Union[str, Any]) -> Dict[str, Any]:
    """
    Processes a single image (path or numpy array) for enrollment.
    - Validates exactly one face is detected.
    - Generates ArcFace embedding.
    - Saves it to the local JSON database under the given person_name.
    
    Returns a status dictionary with keys:
    - success (bool): True if enrolled, False otherwise.
    - message (str): Explanation of the result.
    """
    
    # 1. Input Validation
    if not person_name or not person_name.strip():
        return {"success": False, "message": "Invalid person name."}
        
    person_name = person_name.strip()
    
    # 2. Extract Embedding
    try:
        # enforce_detection=True throws an exception if no face is found
        results = DeepFace.represent(
            img_path=image_path_or_array,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
        
        # Format normalization for different DeepFace versions
        if isinstance(results, dict):
            results = [results]
            
        # 3. Validate exactly ONE face
        num_faces = len(results)
        if num_faces == 0:
            return {"success": False, "message": "No face detected in the image."}
        elif num_faces > 1:
            return {"success": False, "message": f"Multiple faces ({num_faces}) detected. Please upload an image with exactly one face."}
            
        embedding = results[0]["embedding"]
        
        # 4. Save to Database
        if add_embedding_to_person(person_name, embedding):
            return {"success": True, "message": "Successfully enrolled."}
        else:
            return {"success": False, "message": "Failed to save embedding to database."}
            
    except ValueError as e:
        if "Face could not be detected" in str(e):
            return {"success": False, "message": "No face detected in the image."}
        return {"success": False, "message": f"Validation Error: {str(e)}"}
    except Exception as e:
        return {"success": False, "message": f"Unexpected error: {str(e)}"}
