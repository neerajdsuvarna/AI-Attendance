"""
Face Capture Module
Reusable functions for capturing face embeddings from multiple angles
Can be used by both CLI scripts and API endpoints
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
import base64
import json
import sys

# Add common directory to path for GPU_Check import
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
try:
    from GPU_Check import get_onnx_provider
except ImportError:
    # Fallback if GPU_Check not available
    def get_onnx_provider():
        available_providers = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers


# ============================================================================
# QUALITY SETTINGS - GPU vs CPU
# ============================================================================
# Based on Buffalo SCRFD (det_10g.onnx) model specifications:
# - Optimal: 640x640 (standard balance point)
# - GPU practical max: 1024x1024 (efficient higher resolution)
# - Minimum: 320x320 (not recommended, accuracy degrades)
# - Must be divisible by 32 (for stride compatibility)

# CPU settings (current/default) - optimized for performance
DETECT_SIZE_CPU = (640, 640)  # Optimal for CPU - standard balance point
DETECTION_THRESHOLD_CPU = 0.5
QUALITY_CONFIDENCE_CPU = 0.5
NMS_THRESHOLD_CPU = 0.4

# GPU settings (higher quality - more strict) - better for capturing all people
DETECT_SIZE_GPU = (1024, 1024)  # Better quality than 640x640, more efficient than 1280x1280
DETECTION_THRESHOLD_GPU = 0.7  # Higher = fewer false positives, better quality
QUALITY_CONFIDENCE_GPU = 0.6   # Higher = stricter quality checks
NMS_THRESHOLD_GPU = 0.3        # Lower = better duplicate suppression

# Active settings (will be set based on GPU availability)
DETECT_SIZE = DETECT_SIZE_CPU
DETECTION_THRESHOLD = DETECTION_THRESHOLD_CPU
QUALITY_CONFIDENCE = QUALITY_CONFIDENCE_CPU
NMS_THRESHOLD = NMS_THRESHOLD_CPU

def _is_gpu_available():
    """Check if GPU is available and being used"""
    try:
        available_providers = ort.get_available_providers()
        return 'CUDAExecutionProvider' in available_providers
    except:
        return False

def _update_settings_for_hardware():
    """Update detection settings based on available hardware"""
    global DETECTION_THRESHOLD, QUALITY_CONFIDENCE, NMS_THRESHOLD, DETECT_SIZE
    
    if _is_gpu_available():
        DETECT_SIZE = DETECT_SIZE_GPU
        DETECTION_THRESHOLD = DETECTION_THRESHOLD_GPU
        QUALITY_CONFIDENCE = QUALITY_CONFIDENCE_GPU
        NMS_THRESHOLD = NMS_THRESHOLD_GPU
        print(f"[INFO] face_capture.py: GPU detected - Using high-quality settings:")
        print(f"       Resolution: {DETECT_SIZE[0]}x{DETECT_SIZE[1]}")
        print(f"       Detection: {DETECTION_THRESHOLD}, Quality: {QUALITY_CONFIDENCE}, NMS: {NMS_THRESHOLD}")
    else:
        DETECT_SIZE = DETECT_SIZE_CPU
        DETECTION_THRESHOLD = DETECTION_THRESHOLD_CPU
        QUALITY_CONFIDENCE = QUALITY_CONFIDENCE_CPU
        NMS_THRESHOLD = NMS_THRESHOLD_CPU
        print(f"[INFO] face_capture.py: CPU mode - Using standard settings:")
        print(f"       Resolution: {DETECT_SIZE[0]}x{DETECT_SIZE[1]}")
        print(f"       Detection: {DETECTION_THRESHOLD}, Quality: {QUALITY_CONFIDENCE}, NMS: {NMS_THRESHOLD}")

# ============================================================================
# MODEL LOADING (Singleton pattern)
# ============================================================================
_models_loaded = False
_det_session = None
_rec_session = None


def load_models():
    """Load face detection and recognition models with GPU support if available"""
    global _models_loaded, _det_session, _rec_session
    
    if _models_loaded:
        return _det_session, _rec_session
    
    print("Loading face recognition models (face_capture.py)...")
    try:
        # Get the directory where this script is located, then look for buffalo_l folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, "buffalo_l")
        det_model_path = os.path.join(model_dir, "det_10g.onnx")
        rec_model_path = os.path.join(model_dir, "w600k_r50.onnx")
        
        if not os.path.exists(det_model_path):
            raise FileNotFoundError(f"Detection model not found: {det_model_path}")
        if not os.path.exists(rec_model_path):
            raise FileNotFoundError(f"Recognition model not found: {rec_model_path}")
        
        # Get optimal execution providers (GPU if available, CPU as fallback)
        providers = get_onnx_provider()
        
        # Log which hardware will be used
        if 'CUDAExecutionProvider' in providers:
            print("[INFO] face_capture.py: CUDA GPU available - using GPU acceleration")
        else:
            import os
            print(f"[INFO] face_capture.py: Using CPU with {os.cpu_count()} cores")
        
        # Configure session options for better performance
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 0  # Use all CPU cores
        sess_options.inter_op_num_threads = 0
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # CPU provider options (only used if CPU is selected)
        cpu_provider_options = {
            'arena_extend_strategy': 'kSameAsRequested',
            'enable_cpu_mem_arena': True,
        }
        
        # Determine provider options based on selected providers
        provider_options = []
        if 'CUDAExecutionProvider' not in providers:
            provider_options = [cpu_provider_options]
        
        # Load detection model
        _det_session = ort.InferenceSession(
            det_model_path,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options if provider_options else None
        )
        
        # Load recognition model
        _rec_session = ort.InferenceSession(
            rec_model_path,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options if provider_options else None
        )
        
        _models_loaded = True
        actual_providers = _det_session.get_providers()
        print(f"[OK] face_capture.py: Models loaded successfully using: {actual_providers}")
        
        # Update detection settings based on hardware (GPU vs CPU)
        _update_settings_for_hardware()
        
        return _det_session, _rec_session
        
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        raise


# ============================================================================
# IMAGE UTILITIES
# ============================================================================
def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        print(f"[ERROR] Failed to decode base64 image: {e}")
        return None


# ============================================================================
# FACE DETECTION
# ============================================================================
def nms(boxes, scores, threshold=None):
    """Non-Maximum Suppression to remove duplicate detections"""
    if threshold is None:
        threshold = NMS_THRESHOLD
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    
    return keep


def detect_faces(frame, threshold=None):
    """Detect faces in frame"""
    if threshold is None:
        threshold = DETECTION_THRESHOLD
    det_session, _ = load_models()
    
    h, w = frame.shape[:2]
    detect_w, detect_h = DETECT_SIZE
    
    # Resize maintaining aspect ratio
    scale = min(detect_w / w, detect_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Prepare input with padding
    img = np.zeros((detect_h, detect_w, 3), dtype=np.uint8)
    img[:new_h, :new_w] = resized
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
    
    # Run detection
    outputs = det_session.run(None, {det_session.get_inputs()[0].name: img})
    
    # Parse detections
    faces = []
    all_scores = []
    
    for idx, stride in enumerate([8, 16, 32]):
        score_map = outputs[idx]
        bbox_map = outputs[idx + 3]
        
        detect_w, detect_h = DETECT_SIZE
        height, width = detect_h // stride, detect_w // stride
        
        # Generate anchors
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        anchor_centers = np.stack([anchor_centers, anchor_centers], axis=1).reshape((-1, 2))
        
        # Process scores and boxes
        scores = score_map.reshape(-1)
        bboxes = bbox_map.reshape(-1, 4)
        
        # Filter by threshold
        valid_mask = scores > threshold
        valid_scores = scores[valid_mask]
        valid_bboxes = bboxes[valid_mask]
        valid_centers = anchor_centers[valid_mask]
        
        if len(valid_scores) == 0:
            continue
        
        # Decode boxes
        x1 = (valid_centers[:, 0] - valid_bboxes[:, 0] * stride) / scale
        y1 = (valid_centers[:, 1] - valid_bboxes[:, 1] * stride) / scale
        x2 = (valid_centers[:, 0] + valid_bboxes[:, 2] * stride) / scale
        y2 = (valid_centers[:, 1] + valid_bboxes[:, 3] * stride) / scale
        
        # Clip to frame bounds
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)
        
        for i in range(len(valid_scores)):
            faces.append([int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])])
            all_scores.append(float(valid_scores[i]))
    
    # Apply NMS
    if len(faces) > 0:
        keep_indices = nms(faces, all_scores, 0.4)
        faces = [faces[i] + [all_scores[i]] for i in keep_indices]
    
    return faces


# ============================================================================
# FACE ALIGNMENT AND EMBEDDING
# ============================================================================
def align_face(frame, bbox):
    """Align face for better recognition"""
    x1, y1, x2, y2 = bbox[:4]
    
    # Add padding
    h, w = frame.shape[:2]
    face_w, face_h = x2 - x1, y2 - y1
    pad_w, pad_h = int(face_w * 0.3), int(face_h * 0.3)
    
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    face = frame[y1:y2, x1:x2]
    
    if face.size == 0:
        return None
    
    # Ensure minimum size
    if face.shape[0] < 40 or face.shape[1] < 40:
        return None
    
    return face


def get_embedding(frame, bbox):
    """Extract normalized 512-d embedding"""
    _, rec_session = load_models()
    
    face = align_face(frame, bbox)
    
    if face is None:
        return None
    
    try:
        face = cv2.resize(face, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Normalize
        face = (face - 127.5) / 128.0
        face = np.transpose(face, (2, 0, 1))[np.newaxis, :]
        
        # Get embedding
        embedding = rec_session.run(None, {rec_session.get_inputs()[0].name: face})[0][0]
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    except Exception as e:
        print(f"[ERROR] Embedding extraction failed: {e}")
        return None


# ============================================================================
# FACE QUALITY CHECKS
# ============================================================================
def check_face_quality(frame, bbox):
    """Check if face meets quality requirements"""
    x1, y1, x2, y2, conf = bbox
    
    # Check confidence (uses GPU/CPU appropriate threshold)
    if conf < QUALITY_CONFIDENCE:
        return False, "Low confidence"
    
    # Check size
    face_w = x2 - x1
    face_h = y2 - y1
    h, w = frame.shape[:2]
    
    min_size = min(w, h) * 0.1  # At least 10% of frame
    if face_w < min_size or face_h < min_size:
        return False, "Face too small"
    
    # Check position (face should be reasonably centered)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    frame_center_x = w / 2
    frame_center_y = h / 2
    
    max_offset = min(w, h) * 0.3  # Allow 30% offset from center
    if abs(center_x - frame_center_x) > max_offset or abs(center_y - frame_center_y) > max_offset:
        return False, "Face not centered"
    
    # Check aspect ratio
    aspect_ratio = face_w / face_h
    if aspect_ratio < 0.7 or aspect_ratio > 1.3:
        return False, "Poor aspect ratio"
    
    return True, "Quality OK"


# ============================================================================
# PROCESS IMAGES FROM FRONTEND (NEW)
# ============================================================================
def process_images_to_embeddings(images_base64):
    """
    Process multiple base64-encoded images and extract face embeddings
    
    Args:
        images_base64: List of dicts with 'angle' (str) and 'image' (base64 str)
    
    Returns:
        dict with:
            - success: bool
            - embeddings: list of dicts with 'angle', 'embedding' (base64), 'confidence'
            - average_embedding: base64 encoded average embedding
            - message: str
    """
    try:
        # Load models
        load_models()
        
        if not images_base64 or len(images_base64) == 0:
            return {
                "success": False,
                "message": "No images provided",
                "embeddings": [],
                "average_embedding": None
            }
        
        captured_embeddings = []
        
        print(f"[INFO] Processing {len(images_base64)} images...")
        
        for img_data in images_base64:
            angle_name = img_data.get('angle', 'Unknown')
            image_base64 = img_data.get('image', '')
            
            if not image_base64:
                print(f"[WARNING] Skipping {angle_name} - no image data")
                continue
            
            # Convert base64 to OpenCV image
            frame = base64_to_image(image_base64)
            if frame is None:
                print(f"[WARNING] Failed to decode {angle_name} image")
                continue
            
            # Detect faces
            faces = detect_faces(frame)  # Uses GPU/CPU appropriate threshold
            
            if len(faces) == 0:
                print(f"[WARNING] No face detected in {angle_name} image")
                continue
            
            # Get best face (highest confidence)
            faces.sort(key=lambda x: x[4], reverse=True)
            best_face = faces[0]
            
            # Check quality
            quality_ok, quality_msg = check_face_quality(frame, best_face)
            
            if not quality_ok:
                print(f"[WARNING] {angle_name} quality check failed: {quality_msg}")
                continue
            
            # Extract embedding
            embedding = get_embedding(frame, best_face)
            
            if embedding is None:
                print(f"[WARNING] Failed to extract embedding for {angle_name}")
                continue
            
            # Convert embedding to base64
            embedding_bytes = embedding.astype(np.float32).tobytes()
            embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
            
            captured_embeddings.append({
                'angle': angle_name,
                'embedding': embedding_b64,
                'confidence': float(best_face[4])
            })
            
            print(f"[OK] {angle_name} angle processed successfully")
        
        if len(captured_embeddings) == 0:
            return {
                "success": False,
                "message": "No valid faces found in any images",
                "embeddings": [],
                "average_embedding": None
            }
        
        # Calculate average embedding
        all_embeddings = []
        for emb_data in captured_embeddings:
            emb_bytes = base64.b64decode(emb_data['embedding'])
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            all_embeddings.append(emb)
        
        avg_embedding = np.mean(all_embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        # Convert average to base64
        avg_embedding_b64 = base64.b64encode(avg_embedding.astype(np.float32).tobytes()).decode('utf-8')
        
        return {
            "success": True,
            "message": f"Successfully processed {len(captured_embeddings)} angles",
            "embeddings": captured_embeddings,
            "average_embedding": avg_embedding_b64,
            "angles_captured": len(captured_embeddings)
        }
        
    except Exception as e:
        print(f"[ERROR] Image processing failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Processing error: {str(e)}",
            "embeddings": [],
            "average_embedding": None
        }


# ============================================================================
# MULTI-ANGLE CAPTURE (Camera Version - for CLI use)
# ============================================================================
def capture_face_embeddings_from_camera(required_angles=5, show_preview=False):
    """
    Capture face embeddings from multiple angles using camera
    
    Args:
        required_angles: Number of angles to capture (default: 5)
        show_preview: Whether to show OpenCV preview window (default: False for API)
    
    Returns:
        dict with:
            - success: bool
            - embeddings: list of dicts with 'angle', 'embedding' (base64), 'confidence'
            - average_embedding: base64 encoded average embedding
            - message: str
    """
    try:
        # Load models
        load_models()
        
        # Initialize camera
        cap = None
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                ret, frame = cap.read()
                if ret and frame is not None:
                    break
                cap.release()
                cap = None
        
        if cap is None:
            return {
                "success": False,
                "message": "No camera found",
                "embeddings": [],
                "average_embedding": None
            }
        
        angles = [
            {"name": "Front", "instruction": "Face the camera directly"},
            {"name": "Left", "instruction": "Turn your head slightly left"},
            {"name": "Right", "instruction": "Turn your head slightly right"},
            {"name": "Slight Up", "instruction": "Tilt your head slightly up"},
            {"name": "Slight Down", "instruction": "Tilt your head slightly down"},
        ]
        
        captured_embeddings = []
        current_angle_idx = 0
        
        print(f"[INFO] Starting multi-angle capture ({required_angles} angles)")
        
        while current_angle_idx < required_angles and current_angle_idx < len(angles):
            angle_info = angles[current_angle_idx]
            print(f"[INFO] Capturing {angle_info['name']} angle...")
            
            angle_captured = False
            attempts = 0
            max_attempts = 300  # ~10 seconds at 30fps
            good_quality_frames = 0  # Track consecutive good quality frames
            min_good_frames = 30  # Need 1 second of good quality at 30fps
            
            while not angle_captured and attempts < max_attempts:
                ret, frame = cap.read()
                if not ret:
                    attempts += 1
                    continue
                
                attempts += 1
                
                # Detect faces
                faces = detect_faces(frame)  # Uses GPU/CPU appropriate threshold
                
                quality_ok = False
                quality_msg = "No face detected"
                best_face = None
                
                if len(faces) > 0:
                    faces.sort(key=lambda x: x[4], reverse=True)
                    best_face = faces[0]
                    quality_ok, quality_msg = check_face_quality(frame, best_face)
                
                # Show preview if requested
                if show_preview:
                    display = frame.copy()
                    if best_face:
                        color = (0, 255, 0) if quality_ok else (0, 165, 255)
                        x1, y1, x2, y2, conf = best_face
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display, f"Conf: {conf:.2f}", (x1, y1-40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    cv2.putText(display, f"Angle {current_angle_idx + 1}/{required_angles}: {angle_info['name']}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(display, angle_info['instruction'], (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    cv2.putText(display, quality_msg, (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if quality_ok else (0, 165, 255), 2)
                    cv2.putText(display, f"Captured: {len(captured_embeddings)}/{required_angles}", 
                               (10, display.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    cv2.imshow('Face Capture', display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        cap.release()
                        if show_preview:
                            cv2.destroyAllWindows()
                        return {
                            "success": False,
                            "message": "Capture cancelled by user",
                            "embeddings": [],
                            "average_embedding": None
                        }
                    elif key == 32:  # SPACE - manual trigger
                        if quality_ok and best_face:
                            angle_captured = True
                else:
                    # Auto-capture when quality is good (for API mode)
                    # Wait for stable face detection (at least 1 second of good quality)
                    if quality_ok and best_face:
                        good_quality_frames += 1
                        if good_quality_frames >= min_good_frames:
                            angle_captured = True
                    else:
                        good_quality_frames = 0  # Reset counter if quality drops
                
                if angle_captured and best_face:
                    try:
                        embedding = get_embedding(frame, best_face)
                        
                        if embedding is None:
                            print(f"[WARNING] Failed to extract embedding for {angle_info['name']}")
                            angle_captured = False
                            continue
                        
                        # Convert embedding to base64 for JSON serialization
                        embedding_bytes = embedding.astype(np.float32).tobytes()
                        embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
                        
                        captured_embeddings.append({
                            'angle': angle_info['name'],
                            'embedding': embedding_b64,
                            'confidence': float(best_face[4])
                        })
                        
                        print(f"[OK] {angle_info['name']} angle captured!")
                        current_angle_idx += 1
                        break
                        
                    except Exception as e:
                        print(f"[ERROR] Failed to process {angle_info['name']} angle: {e}")
                        angle_captured = False
                        good_quality_frames = 0  # Reset quality counter
                        continue
        
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        if len(captured_embeddings) < required_angles:
            return {
                "success": False,
                "message": f"Only captured {len(captured_embeddings)}/{required_angles} angles",
                "embeddings": captured_embeddings,
                "average_embedding": None
            }
        
        # Calculate average embedding
        all_embeddings = []
        for emb_data in captured_embeddings:
            emb_bytes = base64.b64decode(emb_data['embedding'])
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            all_embeddings.append(emb)
        
        avg_embedding = np.mean(all_embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        # Convert average to base64
        avg_embedding_b64 = base64.b64encode(avg_embedding.astype(np.float32).tobytes()).decode('utf-8')
        
        return {
            "success": True,
            "message": f"Successfully captured {len(captured_embeddings)} angles",
            "embeddings": captured_embeddings,
            "average_embedding": avg_embedding_b64,
            "angles_captured": len(captured_embeddings)
        }
        
    except Exception as e:
        print(f"[ERROR] Capture failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Capture error: {str(e)}",
            "embeddings": [],
            "average_embedding": None
        }
