"""
Improved Face Registration System
Key improvements:
- Proper face alignment matching recognition code
- Embedding normalization (CRITICAL)
- Multiple quality checks
- Visual feedback during registration
- Same preprocessing as recognition system
"""

import cv2
import numpy as np
from setup_database import DatabaseQueries
import onnxruntime as ort
import os

# ============================================================================
# MODEL LOADING
# ============================================================================
print("Loading face detection and recognition models...")

try:
    model_dir = r"E:\AI ATTENDANCE\Attendance\backend\buffalo_l"
    det_model_path = os.path.join(model_dir, "det_10g.onnx")
    rec_model_path = os.path.join(model_dir, "w600k_r50.onnx")
    
    if not os.path.exists(det_model_path):
        print(f"[ERROR] Detection model not found: {det_model_path}")
        exit(1)
    if not os.path.exists(rec_model_path):
        print(f"[ERROR] Recognition model not found: {rec_model_path}")
        exit(1)
    
    det_session = ort.InferenceSession(det_model_path, providers=['CPUExecutionProvider'])
    rec_session = ort.InferenceSession(rec_model_path, providers=['CPUExecutionProvider'])
    
    print(f"[OK] Loaded detection model: {os.path.basename(det_model_path)}")
    print(f"[OK] Loaded recognition model: {os.path.basename(rec_model_path)}")
    
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# ============================================================================
# IMPROVED DETECTION WITH NMS
# ============================================================================
def nms(boxes, scores, threshold=0.4):
    """Non-Maximum Suppression to remove duplicate detections"""
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


def detect_faces(frame, threshold=0.6):
    """
    Improved face detection matching the recognition code
    Returns list of [x1, y1, x2, y2, confidence] in original frame coordinates
    """
    h, w = frame.shape[:2]
    
    # Resize maintaining aspect ratio
    scale = min(640 / w, 640 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Prepare input with padding
    img = np.zeros((640, 640, 3), dtype=np.uint8)
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
        
        height, width = 640 // stride, 640 // stride
        
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
        
        # Decode boxes (distance format)
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
# IMPROVED FACE ALIGNMENT AND EMBEDDING
# ============================================================================
def align_face(frame, bbox):
    """
    Align face - MUST match the recognition code exactly
    """
    x1, y1, x2, y2 = bbox[:4]
    
    # Add padding (same as recognition code)
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
    """
    Extract normalized embedding - MUST match recognition code
    """
    face = align_face(frame, bbox)
    
    if face is None:
        return None
    
    try:
        # Resize to model input size
        face = cv2.resize(face, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Normalize (same as recognition)
        face = (face - 127.5) / 128.0
        face = np.transpose(face, (2, 0, 1))[np.newaxis, :]
        
        # Get embedding
        embedding = rec_session.run(None, {rec_session.get_inputs()[0].name: face})[0][0]
        
        # CRITICAL: Normalize embedding (this was missing!)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    except Exception as e:
        print(f"[ERROR] Embedding extraction failed: {e}")
        return None


# ============================================================================
# QUALITY CHECKS
# ============================================================================
def check_face_quality(frame, bbox):
    """
    Check if face meets quality requirements
    Returns (is_good, message)
    """
    x1, y1, x2, y2 = bbox[:4]
    face_w, face_h = x2 - x1, y2 - y1
    
    # Check size
    if face_w < 80 or face_h < 80:
        return False, "Face too small (move closer)"
    
    if face_w > frame.shape[1] * 0.8:
        return False, "Face too large (move back)"
    
    # Check position (should be centered)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    frame_center_x = frame.shape[1] / 2
    frame_center_y = frame.shape[0] / 2
    
    if abs(center_x - frame_center_x) > frame.shape[1] * 0.3:
        return False, "Center your face horizontally"
    
    if abs(center_y - frame_center_y) > frame.shape[0] * 0.3:
        return False, "Center your face vertically"
    
    # Check aspect ratio (should be roughly square)
    aspect_ratio = face_w / face_h
    if aspect_ratio < 0.7 or aspect_ratio > 1.3:
        return False, "Face angle incorrect (face forward)"
    
    # Check brightness
    face_region = frame[y1:y2, x1:x2]
    if face_region.size > 0:
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 60:
            return False, "Too dark (improve lighting)"
        if mean_brightness > 200:
            return False, "Too bright (reduce lighting)"
    
    return True, "Good quality!"


# ============================================================================
# CAMERA INITIALIZATION
# ============================================================================
def init_camera():
    """Initialize camera with proper settings"""
    print("\n[INFO] Attempting to open webcam...")
    
    for camera_index in [0, 1, 2]:
        print(f"  Trying camera index {camera_index}...")
        
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            # Set higher resolution for better quality
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            ret, frame = cap.read()
            if ret and frame is not None:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  [OK] Camera found at index {camera_index} ({actual_w}x{actual_h})")
                return cap
            else:
                cap.release()
        else:
            cap.release()
    
    return None


# ============================================================================
# MULTI-ANGLE CAPTURE
# ============================================================================
def capture_multiple_angles(cap, required_angles=5):
    """
    Capture face from multiple angles for better recognition
    Returns list of embeddings or None if failed
    """
    angles = [
        {"name": "Front", "instruction": "Face the camera directly"},
        {"name": "Left", "instruction": "Turn your head slightly left"},
        {"name": "Right", "instruction": "Turn your head slightly right"},
        {"name": "Slight Up", "instruction": "Tilt your head slightly up"},
        {"name": "Slight Down", "instruction": "Tilt your head slightly down"},
    ]
    
    captured_embeddings = []
    current_angle_idx = 0
    
    print("\n" + "="*70)
    print("MULTI-ANGLE FACE CAPTURE")
    print("="*70)
    print(f"Capturing {required_angles} angles for better recognition...")
    print("="*70)
    
    while current_angle_idx < required_angles:
        angle_info = angles[current_angle_idx]
        print(f"\n[{current_angle_idx + 1}/{required_angles}] {angle_info['name']} Angle")
        print(f"Instruction: {angle_info['instruction']}")
        print("Press SPACE when ready, ESC to cancel")
        
        angle_captured = False
        
        while not angle_captured:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to capture frame")
                return None
            
            display = frame.copy()
            
            # Detect faces
            faces = detect_faces(frame, threshold=0.5)
            
            quality_ok = False
            quality_msg = "No face detected"
            best_face = None
            
            if len(faces) > 0:
                faces.sort(key=lambda x: x[4], reverse=True)
                best_face = faces[0]
                quality_ok, quality_msg = check_face_quality(frame, best_face)
                
                color = (0, 255, 0) if quality_ok else (0, 165, 255)
                x1, y1, x2, y2, conf = best_face
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"Conf: {conf:.2f}", (x1, y1-40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw angle-specific instructions
            status_color = (0, 255, 0) if quality_ok else (0, 165, 255)
            cv2.putText(display, f"Angle {current_angle_idx + 1}/{required_angles}: {angle_info['name']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, angle_info['instruction'], (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(display, quality_msg, (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            instruction = "Press SPACE to capture this angle" if quality_ok else "Adjust position"
            cv2.putText(display, instruction, (10, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show progress
            progress_text = f"Captured: {len(captured_embeddings)}/{required_angles}"
            cv2.putText(display, progress_text, (10, display.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('Multi-Angle Face Capture [SPACE=Capture, ESC=Cancel]', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE
                if not quality_ok:
                    print(f"[WARNING] Quality check failed: {quality_msg}")
                    print("[INFO] Please adjust and try again")
                    continue
                
                if best_face is None:
                    print("[ERROR] No face detected!")
                    continue
                
                print(f"[INFO] Capturing {angle_info['name']} angle...")
                
                try:
                    embedding = get_embedding(frame, best_face)
                    
                    if embedding is None:
                        print("[ERROR] Failed to extract embedding")
                        continue
                    
                    captured_embeddings.append({
                        'angle': angle_info['name'],
                        'embedding': embedding,
                        'confidence': best_face[4]
                    })
                    
                    print(f"[OK] {angle_info['name']} angle captured!")
                    print(f"     Embedding norm: {np.linalg.norm(embedding):.4f}")
                    
                    angle_captured = True
                    current_angle_idx += 1
                    
                    # Brief pause to show success
                    cv2.putText(display, "✓ Captured!", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow('Multi-Angle Face Capture [SPACE=Capture, ESC=Cancel]', display)
                    cv2.waitKey(1000)  # Show success message for 1 second
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process {angle_info['name']} angle: {e}")
                    continue
            
            elif key == 27:  # ESC
                print("\n[INFO] Multi-angle capture cancelled")
                return None
    
    print(f"\n[SUCCESS] All {required_angles} angles captured!")
    return captured_embeddings


# ============================================================================
# MAIN REGISTRATION LOOP
# ============================================================================
def main():
    print("="*70)
    print("IMPROVED FACE REGISTRATION SYSTEM - MULTI-ANGLE")
    print("="*70)
    print("This system captures faces from multiple angles for better recognition")
    print("="*70)
    
    cap = init_camera()
    
    if cap is None or not cap.isOpened():
        print("\n[ERROR] Cannot open webcam!")
        print("Troubleshooting:")
        print("  1. Close any apps using the camera")
        print("  2. Check camera permissions in Windows Settings")
        print("  3. Try unplugging and replugging USB camera")
        exit(1)
    
    print("\n[INFO] Camera ready. Follow the on-screen instructions.")
    
    capturing = False
    capture_countdown = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break
        
        display = frame.copy()
        
        # Detect faces in real-time
        faces = detect_faces(frame, threshold=0.5)  # Lower threshold for preview
        
        quality_ok = False
        quality_msg = "No face detected"
        best_face = None
        
        if len(faces) > 0:
            # Sort by confidence
            faces.sort(key=lambda x: x[4], reverse=True)
            best_face = faces[0]
            
            x1, y1, x2, y2, conf = best_face
            
            # Check quality
            quality_ok, quality_msg = check_face_quality(frame, best_face)
            
            # Draw rectangle
            color = (0, 255, 0) if quality_ok else (0, 165, 255)  # Green if good, orange if not
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence
            cv2.putText(display, f"Confidence: {conf:.2f}", (x1, y1-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw status message
        status_color = (0, 255, 0) if quality_ok else (0, 165, 255)
        cv2.putText(display, quality_msg, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Draw instructions
        instruction = "Press SPACE to capture" if quality_ok else "Adjust position/lighting"
        cv2.putText(display, instruction, (10, display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show multiple faces warning
        if len(faces) > 1:
            cv2.putText(display, f"WARNING: {len(faces)} faces detected!", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Register Face [SPACE=Capture, ESC=Exit]', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Press SPACE to start multi-angle capture
        if key == 32:  # SPACE key
            if not quality_ok:
                print(f"\n[WARNING] Quality check failed: {quality_msg}")
                print("[INFO] Please adjust and try again")
                continue
            
            if len(faces) > 1:
                print(f"\n[WARNING] Multiple faces detected ({len(faces)})")
                response = input("Continue anyway? (y/n): ").strip().lower()
                if response != 'y':
                    continue
            
            if best_face is None:
                print("\n[ERROR] No face detected!")
                continue
            
            # Start multi-angle capture
            print("\n[INFO] Starting multi-angle face capture...")
            captured_embeddings = capture_multiple_angles(cap, required_angles=5)
            
            if captured_embeddings is None or len(captured_embeddings) == 0:
                print("[ERROR] Failed to capture multiple angles")
                continue
            
            print(f"\n[OK] Successfully captured {len(captured_embeddings)} angles")
            
            # Calculate average embedding (more robust for recognition)
            all_embeddings = np.array([e['embedding'] for e in captured_embeddings])
            avg_embedding = np.mean(all_embeddings, axis=0)
            
            # Normalize the average embedding
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            
            print(f"[OK] Average embedding computed")
            print(f"     Shape: {avg_embedding.shape}")
            print(f"     Norm: {np.linalg.norm(avg_embedding):.4f}")
            
            # Calculate similarity between angles (should be reasonably high)
            print("\n[INFO] Checking angle similarities...")
            similarities = []
            for i in range(len(captured_embeddings)):
                for j in range(i + 1, len(captured_embeddings)):
                    sim = np.dot(captured_embeddings[i]['embedding'], 
                                captured_embeddings[j]['embedding'])
                    similarities.append(sim)
                    print(f"  {captured_embeddings[i]['angle']} vs {captured_embeddings[j]['angle']}: {sim:.4f}")
            
            avg_similarity = np.mean(similarities) if similarities else 0
            print(f"  Average similarity: {avg_similarity:.4f}")
            
            if avg_similarity < 0.7:
                print("[WARNING] Low similarity between angles. Consider recapturing.")
                response = input("Continue anyway? (y/n): ").strip().lower()
                if response != 'y':
                    continue
            
            # Get employee name
            print("\n" + "="*70)
            name = input("Enter employee name: ").strip()
            if not name:
                print("[ERROR] Name cannot be empty")
                continue
            
            # Confirm
            print(f"\nRegistering: {name}")
            print(f"Angles captured: {len(captured_embeddings)}")
            print(f"Average similarity: {avg_similarity:.4f}")
            confirm = input("Confirm registration? (y/n): ").strip().lower()
            
            if confirm != 'y':
                print("[INFO] Registration cancelled")
                continue
            
            # Save to database
            # Store the average embedding as the primary embedding
            # Optionally, we can store all embeddings concatenated
            try:
                with DatabaseQueries() as db:
                    # Store average embedding (primary)
                    employee_id = db.insert_employee(name, avg_embedding.tobytes())
                    
                    if employee_id:
                        print(f"\n[SUCCESS] Employee '{name}' registered!")
                        print(f"[SUCCESS] Employee ID: {employee_id}")
                        print(f"[INFO] Stored average embedding from {len(captured_embeddings)} angles")
                        
                        # Verify saved data
                        emp_data = db.get_employee_by_id(employee_id)
                        if emp_data and emp_data[2]:
                            saved_embedding = np.frombuffer(emp_data[2], dtype=np.float32)
                            print(f"[OK] Verification: Saved embedding shape = {saved_embedding.shape}")
                            
                            # Normalize saved embedding
                            norm = np.linalg.norm(saved_embedding)
                            if norm > 0:
                                saved_embedding = saved_embedding / norm
                            
                            # Check similarity with average (should be ~1.0)
                            similarity = np.dot(avg_embedding, saved_embedding)
                            print(f"[OK] Self-similarity check: {similarity:.4f} (should be ~1.0)")
                            
                            if similarity < 0.95:
                                print("[WARNING] Low self-similarity! Check database encoding.")
                        else:
                            print("[WARNING] Could not verify saved data")
                    else:
                        print("[ERROR] Failed to register employee")
                        
            except Exception as e:
                print(f"[ERROR] Database error: {e}")
                import traceback
                traceback.print_exc()
            
            print("\n" + "="*70)
            another = input("Register another employee? (y/n): ").strip().lower()
            if another != 'y':
                break
        
        # Press ESC to exit
        elif key == 27:  # ESC key
            print("\n[INFO] Registration cancelled")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Registration complete!")


if __name__ == "__main__":
    main()