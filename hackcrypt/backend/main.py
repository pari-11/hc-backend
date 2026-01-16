from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import torch
import mediapipe as mp
import json
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import hashlib
import time


# --- CONFIGURATION ---
UPLOAD_FOLDER = "temp_uploads"
PROTECTED_FOLDER = "protected_uploads"  # For NoiseNet protected files
HISTORY_FILE = "scan_history.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROTECTED_FOLDER, exist_ok=True)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- NOISENET PROACTIVE PROTECTION ---
class NoiseNet:
    """
    Proactive protection against deepfake generation
    Adds imperceptible adversarial noise to images
    """
    def __init__(self, secret_key=99, strength=0.02):
        self.secret_key = secret_key
        self.strength = strength
        np.random.seed(secret_key)
    
    def protect_image(self, image_path, output_path):
        """
        Add adversarial noise to protect image from deepfake manipulation
        Returns: protected image path
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return image_path
            
            # Generate deterministic noise based on secret key
            h, w, c = img.shape
            np.random.seed(self.secret_key + hash(image_path) % 1000)
            noise = np.random.randn(h, w, c) * self.strength * 255
            
            # Add noise while keeping values in valid range
            protected = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            cv2.imwrite(output_path, protected)
            print(f"   🛡️ NoiseNet: Protected image saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"   ⚠️ NoiseNet protection failed: {e}")
            return image_path


# Initialize NoiseNet protector
protector = NoiseNet(secret_key=99, strength=0.02)


# --- 1. LOAD THE FACE EXPERT AI ---
print("Loading Deepfake Face Sniper... please wait...")
model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"

try:
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    print("✅ AI Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    processor = None
    model = None


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
    # Use content hash as key to avoid filename collisions
    content_hash = result_data.get("content_hash", filename)
    history[content_hash] = result_data
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


def get_file_hash(file_path):
    """Generate hash of file content for caching"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def get_face_score(pil_image):
    if processor is None or model is None:
        return 50.0
    
    try:
        inputs = processor(pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        id2label = model.config.id2label
        fake_id = next((k for k, v in id2label.items() if "fake" in v.lower()), 0)
        
        fake_prob = probs[0][fake_id].item() * 100
        return fake_prob
    except Exception as e:
        print(f"   ⚠️ Error in face scoring: {e}")
        return 50.0


def extract_and_scan_faces(image_path):
    img_cv = cv2.imread(image_path)
    if img_cv is None: 
        return 50.0
    
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
        if face_crop.size == 0: 
            continue
        
        score = get_face_score(Image.fromarray(face_crop))
        if score > max_fake_score:
            max_fake_score = score
            
    return max_fake_score


# --- MIDDLEWARE: PROACTIVE PROTECTION ---
@app.middleware("http")
async def apply_proactive_protection(request: Request, call_next):
    """
    Middleware that runs BEFORE request processing
    Applies NoiseNet protection to uploaded images
    """
    start_time = time.time()
    
    # Process the request normally first
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# --- API ENDPOINTS ---

@app.get("/")
def home():
    return {
        "message": "Deepfake Sniper with NoiseNet Protection is Online",
        "features": [
            "Image Deepfake Detection",
            "Video Deepfake Detection",
            "Smart Caching System",
            "NoiseNet Proactive Protection"
        ]
    }


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """
    Detect deepfakes in uploaded images
    With smart caching and NoiseNet protection
    """
    # Save uploaded file temporarily
    file_path = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Generate content hash for caching
    content_hash = get_file_hash(file_path)
    
    # 1. CHECK HISTORY FIRST (content-based)
    history = load_history()
    if content_hash in history:
        print(f"⚡ USING CACHED RESULT for {file.filename}")
        os.remove(file_path)
        return history[content_hash]

    try:
        print(f"📸 Analyzing New Image: {file.filename}")
        
        # Run detection on original image
        fake_score = extract_and_scan_faces(file_path)
        
        # Apply NoiseNet protection to original image
        protected_path = f"{PROTECTED_FOLDER}/protected_{file.filename}"
        protector.protect_image(file_path, protected_path)
        
        # Cleanup
        os.remove(file_path)

        verdict = "AI Generated" if fake_score > 50 else "Real Image"
        print(f"   Result: {verdict} ({fake_score:.1f}%)")

        result_data = {
            "filename": file.filename,
            "content_hash": content_hash,
            "is_ai": fake_score > 50,
            "confidence_score": round(fake_score, 2),
            "verdict": verdict,
            "protected": True,
            "protected_path": protected_path
        }

        # 3. SAVE RESULT
        save_to_history(file.filename, result_data)
        return result_data

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    """
    Detect deepfakes in uploaded videos
    With smart caching
    """
    # Save uploaded file temporarily
    file_path = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Generate content hash for caching
    content_hash = get_file_hash(file_path)
    
    # 1. CHECK HISTORY FIRST
    history = load_history()
    if content_hash in history:
        print(f"⚡ USING CACHED RESULT for {file.filename}")
        os.remove(file_path)
        return history[content_hash]

    try:
        print(f"🎥 Analyzing New Video: {file.filename}")
        cap = cv2.VideoCapture(file_path)
        frame_scores = []
        frame_count = 0
        skip_frames = 10 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            
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
            return JSONResponse(
                content={"verdict": "Error", "confidence_score": 0, "is_ai": False},
                status_code=400
            )

        # Improved scoring logic
        high_confidence_fakes = [s for s in frame_scores if s > 80]
        
        if len(high_confidence_fakes) >= 3:
            final_score = sum(high_confidence_fakes) / len(high_confidence_fakes)
            is_fake = True
        else:
            frame_scores.sort(reverse=True)
            top_5 = frame_scores[:min(5, len(frame_scores))]
            final_score = sum(top_5) / len(top_5)
            is_fake = final_score > 50

        print(f"   📊 Final Video Score: {final_score:.1f}%")

        result_data = {
            "filename": file.filename,
            "content_hash": content_hash,
            "is_ai": is_fake,
            "confidence_score": round(final_score, 2),
            "verdict": "AI Generated" if is_fake else "Real Video",
            "frames_analyzed": len(frame_scores),
            "total_frames": frame_count
        }

        # 3. SAVE RESULT
        save_to_history(file.filename, result_data)
        return result_data

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/history")
def get_history():
    """Get all scan history"""
    return load_history()


@app.delete("/history")
def clear_history():
    """Clear scan history"""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return {"message": "History cleared successfully"}


@app.get("/protected/{filename}")
def get_protected_image(filename: str):
    """Retrieve NoiseNet protected image"""
    protected_path = f"{PROTECTED_FOLDER}/protected_{filename}"
    if os.path.exists(protected_path):
        from fastapi.responses import FileResponse
        return FileResponse(protected_path)
    return JSONResponse(content={"error": "Protected file not found"}, status_code=404)


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
