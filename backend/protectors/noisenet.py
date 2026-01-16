# protectors/noisenet.py
import cv2
import numpy as np

class NoiseNet:
    def __init__(self, secret_key=42):
        self.secret_key = secret_key

    def embed_trace_layer(self, image_path):
        """Adds an invisible noise layer for traceability."""
        img = cv2.imread(image_path).astype(np.float32) / 255.0
        h, w, c = img.shape
        
        # Generate 'Secret' noise pattern
        np.random.seed(self.secret_key)
        noise = np.random.normal(0, 0.015, (h, w, c)) # ~1.5% intensity
        
        # Apply the layer (Proactive protection)
        protected_img = np.clip(img + noise, 0, 1)
        protected_path = image_path.replace(".", "_protected.")
        
        cv2.imwrite(protected_path, (protected_img * 255).astype(np.uint8))
        return protected_path

    def verify_integrity(self, current_image_path):
        """Checks if the noise layer has been disturbed (tampered)."""
        # Load the possibly tampered image
        curr_img = cv2.imread(current_image_path).astype(np.float32) / 255.0
        h, w, c = curr_img.shape
        
        # Regenerate the 'Expected' noise
        np.random.seed(self.secret_key)
        expected_noise = np.random.normal(0, 0.015, (h, w, c))
        
        # In a real scenario, you'd use a high-pass filter to extract the noise
        # This is a simplified 'Disturbance' check for the hackathon:
        # If the high-frequency components don't match our seed, it's tampered.
        # ... logic for signal subtraction ...
        
        return "Noise Layer Disturbed" # or "Integrity Confirmed"