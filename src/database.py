import json
import os
from typing import Dict, List, Any
from src.config import DB_PATH

def load_database() -> Dict[str, Any]:
    """
    Loads the JSON database mapping identities to their embeddings.
    Handles empty or missing database files.
    """
    if not os.path.exists(DB_PATH):
        return {}
        
    try:
        with open(DB_PATH, 'r') as f:
            db = json.load(f)
            # Ensure proper schema
            if not isinstance(db, dict):
                return {}
            return db
    except (json.JSONDecodeError, IOError):
        # Return empty dictionary if the file is malformed or inaccessible
        return {}

def save_database(db: Dict[str, Any]) -> bool:
    """
    Saves the dictionary database back to the JSON file.
    """
    try:
        with open(DB_PATH, 'w') as f:
            json.dump(db, f, indent=4)
        return True
    except IOError as e:
        print(f"Error saving database: {e}")
        return False

def add_embedding_to_person(name: str, embedding: List[float]) -> bool:
    """
    Adds a new embedding for a specific person.
    """
    db = load_database()
    
    if name not in db:
        db[name] = {"embeddings": []}
        
    db[name]["embeddings"].append(embedding)
    
    return save_database(db)

def get_all_enrolled_people() -> List[str]:
    db = load_database()
    return list(db.keys())

def get_database_stats() -> Dict[str, int]:
    """
    Returns statistics about the database.
    """
    db = load_database()
    num_people = len(db)
    num_embeddings = sum(len(person_data.get("embeddings", [])) for person_data in db.values())
    
    return {
        "num_people": num_people,
        "num_embeddings": num_embeddings
    }
