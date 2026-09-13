import os
import json
from src.config import DB_PATH, DEFAULT_THRESHOLD
from src.enroll import process_and_enroll_image
from src.identify import identify_face
from src.database import load_database, get_database_stats

def main():
    print("--- STARTING DEMO VALIDATION ---")
    
    # 1. Clear existing database
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Cleared existing database at {DB_PATH}")
        
    # 2. Enroll Person 1
    p1_dir = "data/enrollment/person1"
    p1_images = ["photo1.jpeg", "photo2.jpeg", "photo3.jpeg"]
    print("\n--- Enrolling Person 1 ---")
    for img in p1_images:
        path = os.path.join(p1_dir, img)
        res = process_and_enroll_image("person1", path)
        print(f"Enroll {img}: {res['success']} - {res['message']}")
        
    # 3. Enroll Person 2
    p2_dir = "data/enrollment/person2"
    p2_images = ["p1.jpeg", "p2.jpeg", "p3.jpeg"]
    print("\n--- Enrolling Person 2 ---")
    for img in p2_images:
        path = os.path.join(p2_dir, img)
        res = process_and_enroll_image("person2", path)
        print(f"Enroll {img}: {res['success']} - {res['message']}")
        
    # 4. Verify Database
    stats = get_database_stats()
    db = load_database()
    print("\n--- Database Stats ---")
    print(f"Total People: {stats['num_people']}")
    print(f"Total Embeddings: {stats['num_embeddings']}")
    for person, data in db.items():
        print(f"  {person}: {len(data['embeddings'])} embeddings")
        
    # 5. Identification Test - Person 1
    print("\n--- Identification Test: Person 1 ---")
    p1_test_img = os.path.join(p1_dir, "photo1.jpeg")
    res1 = identify_face(p1_test_img)
    print(f"Testing with: {p1_test_img}")
    print(f"Status: {res1.get('status')}")
    print(f"Identity: {res1.get('identity')}")
    print(f"Similarity: {res1.get('similarity')}")
    print(f"Threshold: {res1.get('threshold')}")

    # 6. Identification Test - Person 2
    print("\n--- Identification Test: Person 2 ---")
    p2_test_img = os.path.join(p2_dir, "p1.jpeg")
    res2 = identify_face(p2_test_img)
    print(f"Testing with: {p2_test_img}")
    print(f"Status: {res2.get('status')}")
    print(f"Identity: {res2.get('identity')}")
    print(f"Similarity: {res2.get('similarity')}")
    print(f"Threshold: {res2.get('threshold')}")

if __name__ == '__main__':
    main()
