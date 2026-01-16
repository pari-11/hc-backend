import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import both analysis modules
from services.image_forensics import full_image_forensics
from services.face_detector import analyze_image_deepfake

def analyze_image_complete(image_path):
    """
    Complete image analysis combining:
    1. Face Detector (AI model predictions)
    2. Image Forensics (ELA, frequency, watermark, metadata)
    
    Returns unified result matching frontend UI
    """
    if not os.path.exists(image_path):
        return {"error": "File not found"}
    
    print(f"\n🔍 Running complete image analysis...")
    print(f"📁 File: {os.path.basename(image_path)}\n")
    
    # 1. Get Face Detector results (AI prediction)
    print("   [1/2] Running AI Face Detector...")
    face_result = analyze_image_deepfake(image_path)
    
    # 2. Get Image Forensics results (ELA, metadata, watermark)
    print("   [2/2] Running Image Forensics...")
    forensics_result = full_image_forensics(image_path)
    
    # Extract key metrics
    ai_confidence = face_result.get("overall_confidence", 0)  # 0-100
    ai_is_fake = face_result.get("is_fake", False)
    
    forensics_fake_count = forensics_result.get("fake_indicators_count", 0)
    forensics_is_fake = forensics_result.get("is_fake", False)
    
    # Weighted decision logic
    # AI model gets 60% weight, forensics gets 40% weight
    ai_weight = 0.6
    forensics_weight = 0.4
    
    # Convert forensics to confidence score (0-100)
    # If forensics says fake, confidence is lower
    if forensics_is_fake:
        forensics_confidence = max(0, 100 - (forensics_fake_count * 30))
    else:
        forensics_confidence = 90  # High confidence if forensics says real
    
    # Combined confidence (0-100 scale)
    combined_confidence = (
        ai_confidence * ai_weight + 
        forensics_confidence * forensics_weight
    )
    
    # Final verdict
    if combined_confidence < 30:
        final_verdict = "AI Generated"
        threat_level = "HIGH"
        is_fake = True
    elif combined_confidence < 50:
        final_verdict = "Likely Manipulated"
        threat_level = "MEDIUM"
        is_fake = True
    elif combined_confidence < 70:
        final_verdict = "Suspicious"
        threat_level = "MEDIUM"
        is_fake = False
    else:
        final_verdict = "Likely Authentic"
        threat_level = "LOW"
        is_fake = False
    
    # Build unified response matching UI
    return {
        # Overview section
        "verdict": final_verdict,
        "overall_confidence": round(combined_confidence / 100, 4),  # 0-1 scale for UI
        "is_fake": is_fake,
        "threat_level": threat_level,
        "classification": face_result.get("classification", "Unknown"),
        "likely_tools": face_result.get("likely_tools", []),
        
        # Confidence breakdown (for UI bars)
        "confidence_breakdown": {
            "Visual": face_result.get("confidence_breakdown", {}).get("Visual", 0),
            "Audio": 0,  # Not applicable for images
            "Temporal": 0,  # Not applicable for images
            "Lip-Sync": 0,  # Not applicable for images
            "Metadata": forensics_result.get("watermark_analysis", {}).get("is_fake", False) and 62 or 0
        },
        
        # Primary findings (from face detector)
        "primary_findings": face_result.get("primary_findings", []),
        
        # Visual Analysis tab (from forensics)
        "visual_analysis": {
            "heatmap_base64": forensics_result.get("heatmap_base64", ""),
            "heatmap_description": forensics_result.get("heatmap_description", ""),
            "detected_artifacts": forensics_result.get("detected_artifacts", []),
            "model_performance": forensics_result.get("model_performance", {})
        },
        
        # Forensics details
        "forensics_analysis": {
            "ela_analysis": forensics_result.get("ela_analysis", {}),
            "frequency_analysis": forensics_result.get("frequency_analysis", {}),
            "watermark_analysis": forensics_result.get("watermark_analysis", {})
        },
        
        # Raw results for debugging
        "_debug": {
            "ai_confidence": ai_confidence,
            "forensics_confidence": forensics_confidence,
            "ai_is_fake": ai_is_fake,
            "forensics_is_fake": forensics_is_fake,
            "combined_confidence": combined_confidence
        }
    }

# === STANDALONE TESTING ===
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    
    print("\n🖼️  COMPLETE IMAGE ANALYSIS")
    print("="*60)
    print("Select an IMAGE file to analyze...")
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    
    if file_path:
        result = analyze_image_complete(file_path)
        
        print("\n" + "="*60)
        print("COMPLETE IMAGE ANALYSIS RESULTS")
        print("="*60)
        
        # Pretty print without heatmap
        display_result = result.copy()
        if "visual_analysis" in display_result and "heatmap_base64" in display_result["visual_analysis"]:
            display_result["visual_analysis"]["heatmap_base64"] = f"[BASE64_DATA_PRESENT]"
        
        print(json.dumps(display_result, indent=2))
        print("="*60)
        
        print(f"\n✅ Final Verdict: {result['verdict']}")
        print(f"📊 Combined Confidence: {result['overall_confidence']*100:.2f}%")
        print(f"⚠️  Threat Level: {result['threat_level']}")
        print(f"🎯 Is Fake: {result['is_fake']}")
        
        if "_debug" in result:
            print(f"\n🔧 Debug Info:")
            print(f"   AI Confidence: {result['_debug']['ai_confidence']:.2f}%")
            print(f"   Forensics Confidence: {result['_debug']['forensics_confidence']:.2f}%")
    else:
        print("❌ No file selected.")
