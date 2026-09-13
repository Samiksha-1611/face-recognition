import os
import json
from src.identify import identify_face
from src.database import get_database_stats

def main():
    print("--- STARTING OUT-OF-SAMPLE TEST VALIDATION ---")
    
    # Verify Database
    stats = get_database_stats()
    print(f"Database Stats: Total People: {stats['num_people']}, Total Embeddings: {stats['num_embeddings']}")

    test_cases = [
        "data/test/person1_test.jpeg",
        "data/test/person2_test.jpeg",
        "data/test/unknown_test.jpeg"
    ]

    for test_img in test_cases:
        if not os.path.exists(test_img):
            print(f"ERROR: {test_img} not found.")
            continue
            
        print(f"\n--- Testing with: {test_img} ---")
        res = identify_face(test_img)
        print(f"Status: {res.get('status')}")
        print(f"Identity: {res.get('identity')}")
        print(f"Similarity: {res.get('similarity')}")
        print(f"Threshold: {res.get('threshold')}")
        if 'message' in res:
             print(f"Message: {res.get('message')}")

if __name__ == '__main__':
    main()
