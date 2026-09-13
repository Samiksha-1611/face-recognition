import os
import json
import numpy as np
from src.identify import compute_cosine_similarity
from src.database import load_database
from deepface import DeepFace
from src.config import MODEL_NAME, DETECTOR_BACKEND

def get_embedding(img_path):
    results = DeepFace.represent(img_path=img_path, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, enforce_detection=True)
    if isinstance(results, dict):
        results = [results]
    return np.array(results[0]["embedding"])

def main():
    db = load_database()
    
    # Test images
    p1_test_path = "data/test/person1_test.jpeg"
    p2_test_path = "data/test/person2_test.jpeg"
    unknown_test_path = "data/test/unknown_test.jpeg"
    
    # Extract embeddings for test images
    print("Extracting test embeddings...")
    p1_test_emb = get_embedding(p1_test_path)
    p2_test_emb = get_embedding(p2_test_path)
    unk_test_emb = get_embedding(unknown_test_path)
    
    genuine_scores = []
    impostor_scores = []
    
    print("\n--- Genuine Comparisons ---")
    # p1 test vs p1 enrollments
    for emb in db['person1']['embeddings']:
        sim = compute_cosine_similarity(p1_test_emb, np.array(emb))
        genuine_scores.append(sim)
        print(f"person1_test vs person1_enroll: {sim}")
        
    # p2 test vs p2 enrollments
    for emb in db['person2']['embeddings']:
        sim = compute_cosine_similarity(p2_test_emb, np.array(emb))
        genuine_scores.append(sim)
        print(f"person2_test vs person2_enroll: {sim}")
        
    print("\n--- Impostor Comparisons ---")
    # p1 test vs p2 enrollments
    for emb in db['person2']['embeddings']:
        sim = compute_cosine_similarity(p1_test_emb, np.array(emb))
        impostor_scores.append(sim)
        print(f"person1_test vs person2_enroll: {sim}")
        
    # p2 test vs p1 enrollments
    for emb in db['person1']['embeddings']:
        sim = compute_cosine_similarity(p2_test_emb, np.array(emb))
        impostor_scores.append(sim)
        print(f"person2_test vs person1_enroll: {sim}")
        
    # unk test vs p1 enrollments
    for emb in db['person1']['embeddings']:
        sim = compute_cosine_similarity(unk_test_emb, np.array(emb))
        impostor_scores.append(sim)
        print(f"unknown_test vs person1_enroll: {sim}")
        
    # unk test vs p2 enrollments
    for emb in db['person2']['embeddings']:
        sim = compute_cosine_similarity(unk_test_emb, np.array(emb))
        impostor_scores.append(sim)
        print(f"unknown_test vs person2_enroll: {sim}")
        
    thresholds = [0.30, 0.35, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65]
    
    print("\n--- Threshold Evaluation ---")
    best_f1 = -1
    best_t = None
    
    for t in thresholds:
        tp = sum(1 for s in genuine_scores if s >= t)
        fn = sum(1 for s in genuine_scores if s < t)
        fp = sum(1 for s in impostor_scores if s >= t)
        tn = sum(1 for s in impostor_scores if s < t)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Threshold: {t:.2f} | TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn} | Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        
        # In case of tie, we want a threshold that provides a balance or strictly rejects impostors
        # We'll just track the best F1. If multiple have the same F1, we'll see it in the print out.
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    print(f"\nBest Threshold (by Max F1): {best_t} with F1: {best_f1}")

if __name__ == '__main__':
    main()
