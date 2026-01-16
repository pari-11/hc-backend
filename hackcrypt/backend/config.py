import os

# Directories
UPLOAD_FOLDER = "temp_uploads"
HISTORY_FILE = "scan_history.json"
CELEB_FACES_DIR = "models/celebrity_faces"

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CELEB_FACES_DIR, exist_ok=True)

# Model Configuration
DEEPFAKE_MODEL = "prithivMLmods/Deep-Fake-Detector-v2-Model"

# Detection Thresholds
BLINK_RATE_MIN = 3     # BPM
BLINK_RATE_MAX = 85    # BPM
AUDIO_CUTOFF_FREQ = 16000  # Hz
METADATA_EDIT_GAP = 1800   # 30 minutes in seconds

# ELA Configuration
ELA_JPEG_QUALITY = 90
ELA_SCALE_FACTOR = 10

# Viral Score Weights
VIRAL_WEIGHTS = {
    "is_hd": 10,
    "has_keywords": 20,
    "short_duration": 10,
}

VIRAL_KEYWORDS = ["breaking", "leaked", "scandal", "exclusive", "urgent", "alert"]

# Watermark Keywords
WATERMARK_KEYWORDS = [
    "generated", "imagined", "midjourney", "dall-e", "bing",
    "creator", "unity", "artificial", "intelligence", "openai", "stock",
    "gemini", "google", "ai", "created with", "made with"  # Added gemini
]


# Tesseract Path (Windows - adjust for your system)
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
