# Face Recognition Identification System

## 🔴 Live Deployment
You can access the live working prototype hosted on Streamlit Community Cloud here:
 **[Launch Application](https://face-recognition-njq655zppft7ku2pgehzny.streamlit.app/)**

## 1. Project Overview
This project is a complete local Face Recognition Identification System developed for an internship assignment. The system allows users to enroll known individuals using face images, securely generates and stores biometric face embeddings, and can identify new query faces in real-time. If a scanned face does not match any enrolled identity above a strict confidence threshold, the system securely rejects it as `UNKNOWN`.

## 2. System Workflow
The biometric pipeline follows a strict, sequential process:
```text
[ Input Image (Upload or Camera) ]
                ↓
      [ Face Detection ] (Locate & align face)
                ↓
    [ Face Embedding ] (Generate 512-d vector)
                ↓
[ Similarity Comparison ] (Against all database embeddings)
                ↓
         [ Best Match ] (Highest similarity score)
                ↓
    [ Threshold Decision ] (Score >= Threshold ?)
                ↓
     [ MATCHED / UNKNOWN ] (Final Output)
```

## 3. Technologies Used
- **Python 3.9+** (Virtual Environment)
- **Streamlit:** Powers the responsive, biometric console frontend.
- **DeepFace:** A high-level library wrapping state-of-the-art recognition architectures.
- **ArcFace:** The primary deep learning model used for generating highly separable embeddings.
- **RetinaFace:** The primary computer vision model used for robust face detection and alignment.
- **Cosine Similarity:** Mathematical approach used to compare the multidimensional feature vectors.
- **Local JSON Storage:** A lightweight, explainable persistence mechanism (`database.json`) storing solely metadata and float arrays.

## 4. Model Used
- **Recognition Model:** **ArcFace**. ArcFace (Additive Angular Margin Loss) was chosen because it produces highly discriminative face embeddings. It maps faces into a 512-dimensional hypersphere where images of the same person are clustered tightly together.
- **Face Detector:** **RetinaFace**. RetinaFace was chosen over older cascades (like Haar or MTCNN) for its superior handling of varying lighting conditions, partial occlusions, and facial alignment via 5-point landmarks.
- **Embeddings Approach:** Rather than storing raw images, the system passes the cropped face through the ArcFace ResNet architecture to extract a mathematical representation (a 512-dimensional vector). This is computationally efficient, scalable, and preserves user privacy.

## 5. Matching Method
To identify a face, the system computes the **Cosine Similarity** between the queried face's 512-d embedding and every embedding stored in the database. Cosine similarity measures the angle between two vectors, resulting in a score bounded between `0.0` and `1.0`.

**The UNKNOWN Rule:**
The system uses a strict rejection mechanism rather than simply guessing the highest match:
- `Best Similarity >= 0.45` → **MATCHED** (Identity Accepted)
- `Best Similarity < 0.45`  → **UNKNOWN** (Identity Rejected)

## 6. Enrollment
A new identity is registered in the system via the `ENROLL PERSON` interface. The system supports multi-image enrollment (2-3 images recommended). For each uploaded image:
1. RetinaFace verifies that exactly one clear face exists.
2. ArcFace generates a unique 512-d embedding.
3. The embedding is appended to the identity's array in `database.json`.
Storing multiple variations (e.g., different lighting or angles) greatly increases matching robustness.

## 7. Identification
When a user uploads a photo or captures a live webcam feed on the `IDENTIFY FACE` page:
1. A single query embedding is generated.
2. The system iterates through the entire enrolled database.
3. It computes the cosine similarity against *all* stored embeddings for all identities.
4. It groups the scores by identity and selects the highest score per identity (the best candidate).
5. The highest overall score is then subjected to the threshold logic.

## 8. UNKNOWN Rejection
Without a threshold, a face recognition model will mathematically always find a "closest match"—even if the person is a total stranger. The UNKNOWN rejection mechanism is a critical security feature. The calibrated threshold (0.45) ensures that the system firmly rejects faces that are too far from the enrolled mathematical clusters, preventing unauthorized access.

## 9. Evaluation
The assignment requires basic evaluation results. Due to the scope of this prototype, we ran a targeted local evaluation directly on the provided demo dataset rather than downloading massive external databases (like LFW). 

- **Evaluation Methodology:** A standalone script (`evaluation/evaluate.py`) computes cosine similarities for a matrix of local test images.
- **Sample Size:** 16 pairs (8 Genuine, 8 Impostor) utilizing 9 unique local images.
- **Genuine Comparisons (Same Person):** The similarities ranged from `~0.495` to `~0.51`.
- **Impostor Comparisons (Different People / Strangers):** The similarities ranged from `~0.10` to `~0.392`.
- **Threshold Calibration:** Based on these real-world observed distances, the application threshold was strictly calibrated to **0.45**. This value successfully maximizes the margin, correctly rejecting the highest impostor (`0.392`) while accepting the lowest genuine match (`0.495`).

*Note: This is a small demonstration evaluation specifically designed to prove the pipeline's logic and threshold calibration methodology. It should not be extrapolated as production-level accuracy for millions of users.*

## 10. Failure Cases / Limitations
- **Extreme Poses:** While RetinaFace handles slight profiles, full 90-degree profile shots will fail detection.
- **Poor Lighting / Low Resolution:** Severely degraded images result in noisy embeddings, lowering the cosine similarity and causing False Rejects.
- **Twins / Doppelgangers:** The system relies entirely on 2D geometric features and may struggle with identical twins.
- **Liveness:** The prototype lacks anti-spoofing; it can be fooled by presenting a static photograph to the camera.
- **Small Dataset:** The 0.45 threshold is highly effective for this specific demo environment, but a much larger, statistically significant dataset is required to establish a robust threshold for global deployment.

## 11. Improvements
Future iterations of this system should implement:
1. **Liveness Detection:** Integrating a 3D depth sensor or anti-spoofing neural net (e.g., MiniFASNet).
2. **Vector Database Migration:** Replacing the JSON file with FAISS or Milvus for sub-millisecond similarity searches at scale.
3. **Quality Filtering:** Rejecting blurred or severely backlit images prior to embedding generation.
4. **Large-Scale Evaluation:** Running the pipeline against the full 13,000+ Labeled Faces in the Wild (LFW) dataset to calculate an enterprise-grade threshold.
5. **Continuous WebRTC Stream:** Upgrading the Streamlit camera to a WebRTC live-video feed for continuous tracking.

## 12. Application Usage

### Installation
Ensure you have Python 3.9 or higher installed.

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### How to Run
Start the biometric console by running:
```bash
streamlit run app.py
```

### How to Use the System
1. **IDENTIFY FACE (Default):** Select either `[UPLOAD IMAGE]` or `[LIVE CAMERA]`. Submit a face. The system will overlay a scanner animation while processing the embedding, and then return a MATCH / NO MATCH biometric card.
2. **ENROLL PERSON:** Enter a name and upload 2-3 clear reference photos. The system will process them and store the embeddings in the database.
3. **DATABASE:** View a live summary of all enrolled identities and their respective embedding counts.
