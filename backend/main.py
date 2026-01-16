from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
import cv2
import torch
import mediapipe as mp
import json  # Added for saving history
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --- CONFIGURATION ---
UPLOAD_FOLDER = "temp_uploads"
HISTORY_FILE = "scan_history.json"  # File to store past results
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI()

# --- 1. LOAD THE FACE EXPERT AI ---
print("Loading Deepfake Face Sniper... please wait...")
model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"

try:
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    print("✅ AI Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- 2. INITIALIZE MEDIAPIPE ---
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# --- HELPER FUNCTIONS ---

def load_history():
    """Loads the history of scanned files."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_to_history(filename, result_data):
    """Saves a new scan result to the history file."""
    history = load_history()
    history[filename] = result_data
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def get_face_score(pil_image):
    inputs = processor(pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    id2label = model.config.id2label
    fake_id = next((k for k, v in id2label.items() if "fake" in v.lower()), 0)
    
    fake_prob = probs[0][fake_id].item() * 100
    return fake_prob

def extract_and_scan_faces(image_path):
    img_cv = cv2.imread(image_path)
    if img_cv is None: return 0.0
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    
    if not results.detections:
        print("   ⚠️ No face detected. Scanning full image.")
        return get_face_score(Image.fromarray(img_rgb))
    
    max_fake_score = 0.0
    h, w, _ = img_cv.shape

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        
        padding = int(bw * 0.2) 
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)
        
        face_crop = img_rgb[y1:y2, x1:x2]
        if face_crop.size == 0: continue
        
        score = get_face_score(Image.fromarray(face_crop))
        if score > max_fake_score:
            max_fake_score = score
            
    return max_fake_score

# --- API ENDPOINTS ---

@app.get("/")
def home():
    return {"message": "Deepfake Sniper (With Caching) is Online."}

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    # 1. CHECK HISTORY FIRST
    history = load_history()
    if file.filename in history:
        print(f"⚡ USING CACHED RESULT for {file.filename}")
        return history[file.filename]

    # 2. IF NOT IN HISTORY, PROCESS IT
    file_path = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        print(f"📸 Analyzing New Image: {file.filename}")
        fake_score = extract_and_scan_faces(file_path)
        os.remove(file_path)

        verdict = "AI Generated" if fake_score > 50 else "Real Image"
        print(f"   Result: {verdict} ({fake_score:.1f}%)")

        result_data = {
            "filename": file.filename,
            "is_ai": fake_score > 50,
            "confidence_score": round(fake_score, 2),
            "verdict": verdict
        }

        # 3. SAVE RESULT
        save_to_history(file.filename, result_data)
        return result_data

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    # 1. CHECK HISTORY FIRST
    history = load_history()
    if file.filename in history:
        print(f"⚡ USING CACHED RESULT for {file.filename}")
        return history[file.filename]

    # 2. IF NOT IN HISTORY, PROCESS IT
    file_path = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        print(f"🎥 Analyzing New Video: {file.filename}")
        cap = cv2.VideoCapture(file_path)
        frame_scores = []
        frame_count = 0
        skip_frames = 10 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            if frame_count % skip_frames == 0:
                temp_frame_path = f"{UPLOAD_FOLDER}/temp_frame.jpg"
                cv2.imwrite(temp_frame_path, frame)
                score = extract_and_scan_faces(temp_frame_path)
                frame_scores.append(score)
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)

        cap.release()
        os.remove(file_path)

        if not frame_scores:
            return {"verdict": "Error", "confidence_score": 0, "is_ai": False}

        high_confidence_fakes = [s for s in frame_scores if s > 80]
        
        if len(high_confidence_fakes) >= 3:
            final_score = sum(high_confidence_fakes) / len(high_confidence_fakes)
            is_fake = True
        else:
            frame_scores.sort(reverse=True)
            top_5 = frame_scores[:5]
            final_score = sum(top_5) / len(top_5)
            is_fake = final_score > 50

        print(f"   📊 Final Video Score: {final_score:.1f}%")

        result_data = {
            "filename": file.filename,
            "is_ai": is_fake,
            "confidence_score": round(final_score, 2),
            "verdict": "AI Generated" if is_fake else "Real Video"
        }

        # 3. SAVE RESULT
        save_to_history(file.filename, result_data)
        return result_data

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)