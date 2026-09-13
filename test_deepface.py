import sys
import argparse

# Force UTF-8 encoding for standard outputs to prevent crashes on Windows terminals
# when DeepFace tries to print emojis (like download links).
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from deepface import DeepFace

def test_pipeline(image_path):
    print(f"Loading image: {image_path}")
    print("Running face detection (RetinaFace) and embedding (ArcFace)...")
    
    try:
        # Enforce detection is True by default. If it can't find a face, it raises ValueError.
        results = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )
        
        # DeepFace.represent returns a list of dictionaries for each detected face in newer versions
        if isinstance(results, dict):
            # Fallback for older deepface versions
            results = [results]
            
        num_faces = len(results)
        print("Detection succeeded!")
        print(f"Number of detected faces: {num_faces}")
        
        if num_faces != 1:
            print(f"Error: Exactly one face was expected, but {num_faces} faces were found.")
            sys.exit(1)
            
        embedding = results[0]["embedding"]
        print(f"Embedding dimension: {len(embedding)}")
        # Format a small preview of the embedding array
        preview = ", ".join([f"{x:.4f}" for x in embedding[:5]])
        print(f"Embedding preview: [{preview}, ...]")
        
    except ValueError as e:
        print("Detection failed.")
        print("Number of detected faces: 0")
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Face Detection and Embedding")
    parser.add_argument("image_path", help="Path to the test image")
    args = parser.parse_args()
    
    test_pipeline(args.image_path)