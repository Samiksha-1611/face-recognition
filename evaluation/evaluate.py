import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from deepface import DeepFace
from src.config import MODEL_NAME, DETECTOR_BACKEND, EVAL_DIR
from src.identify import compute_cosine_similarity

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.makedirs(EVAL_DIR, exist_ok=True)

def get_embedding(img_path):
    try:
        results = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
        if isinstance(results, dict):
            results = [results]
        return np.array(results[0]["embedding"])
    except Exception as e:
        print(f"Error extracting embedding for {img_path}: {e}")
        return None

def run_evaluation():
    person1_imgs = [
        "data/enrollment/person1/photo1.jpeg",
        "data/enrollment/person1/photo2.jpeg",
        "data/enrollment/person1/photo3.jpeg",
        "data/test/person1_test.jpeg"
    ]
    person2_imgs = [
        "data/enrollment/person2/p1.jpeg",
        "data/enrollment/person2/p2.jpeg",
        "data/enrollment/person2/p3.jpeg",
        "data/test/person2_test.jpeg"
    ]
    unknown_img = "data/test/unknown_test.jpeg"
    
    all_imgs = person1_imgs + person2_imgs + [unknown_img]
    
    print("Extracting embeddings for all unique images...")
    embeddings_cache = {}
    for img in all_imgs:
        if not os.path.exists(img):
            print(f"Warning: {img} not found.")
            continue
        emb = get_embedding(img)
        if emb is not None:
            embeddings_cache[img] = emb
            
    print(f"Successfully extracted {len(embeddings_cache)} embeddings.")

    pairs = []
    pairs.append((person1_imgs[0], person1_imgs[1], 1))
    pairs.append((person1_imgs[0], person1_imgs[2], 1))
    pairs.append((person1_imgs[1], person1_imgs[3], 1))
    pairs.append((person1_imgs[2], person1_imgs[3], 1))
    pairs.append((person2_imgs[0], person2_imgs[1], 1))
    pairs.append((person2_imgs[0], person2_imgs[2], 1))
    pairs.append((person2_imgs[1], person2_imgs[3], 1))
    pairs.append((person2_imgs[2], person2_imgs[3], 1))
    pairs.append((person1_imgs[0], person2_imgs[0], 0))
    pairs.append((person1_imgs[1], person2_imgs[1], 0))
    pairs.append((person1_imgs[2], person2_imgs[2], 0))
    pairs.append((person1_imgs[3], person2_imgs[3], 0))
    pairs.append((person1_imgs[0], unknown_img, 0))
    pairs.append((person1_imgs[3], unknown_img, 0))
    pairs.append((person2_imgs[0], unknown_img, 0))
    pairs.append((person2_imgs[3], unknown_img, 0))
    
    similarities = []
    labels = []
    
    print(f"Evaluating {len(pairs)} local dataset pairs using {MODEL_NAME} pipeline...")
    for (img1_path, img2_path, is_genuine) in pairs:
        emb1 = embeddings_cache.get(img1_path)
        emb2 = embeddings_cache.get(img2_path)
        if emb1 is not None and emb2 is not None:
            sim = compute_cosine_similarity(emb1, emb2)
            similarities.append(sim)
            labels.append(is_genuine)
            
    if not similarities:
        print("No valid similarities computed.")
        return
        
    df = pd.DataFrame({"similarity": similarities, "is_genuine": labels})
    df.to_csv(os.path.join(EVAL_DIR, "evaluation_results.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    plt.hist(df[df["is_genuine"] == 1]["similarity"], bins=10, alpha=0.5, label='Genuine (Same Person)', color='blue')
    plt.hist(df[df["is_genuine"] == 0]["similarity"], bins=10, alpha=0.5, label='Impostor (Different People)', color='red')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution: Genuine vs Impostor (Local Demo Dataset)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(EVAL_DIR, "similarity_distribution.png"))
    plt.close()
    
    thresholds = np.arange(0.1, 1.0, 0.01)
    best_f1 = 0
    best_thresh = 0
    metrics = []
    
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in df["similarity"]]
        acc = accuracy_score(df["is_genuine"], preds)
        p = precision_score(df["is_genuine"], preds, zero_division=0)
        r = recall_score(df["is_genuine"], preds, zero_division=0)
        f1 = f1_score(df["is_genuine"], preds, zero_division=0)
        metrics.append({"threshold": float(t), "accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)})
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    results = {
        "best_threshold": float(best_thresh),
        "best_f1": float(best_f1),
        "metrics_by_threshold": metrics,
        "total_pairs_evaluated": len(df)
    }
    
    with open(os.path.join(EVAL_DIR, "metrics.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nEvaluation Complete!")
    print(f"Total valid pairs evaluated: {len(df)}")
    print(f"Recommended Threshold (Max F1): {best_thresh:.2f}")
    print(f"Plots and metrics saved to: {EVAL_DIR}")

if __name__ == "__main__":
    run_evaluation()
