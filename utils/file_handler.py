import os
import shutil
from fastapi import UploadFile
from config import UPLOAD_FOLDER

def save_upload(file: UploadFile) -> str:
    """Save uploaded file and return path"""
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path

def cleanup_file(file_path: str):
    """Delete temporary file"""
    if os.path.exists(file_path):
        os.remove(file_path)
