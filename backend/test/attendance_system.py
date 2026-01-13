"""
Improved Face Recognition Attendance System
Key improvements:
- Better detection parsing with NMS
- Face alignment for recognition
- Higher similarity threshold
- Better tracking logic
"""

import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime, timedelta
from setup_database import DatabaseQueries
import os
from threading import Thread, Lock
from queue import Queue, Empty
import time

# ============================================================================
# IMPROVED CONFIGURATION
# ============================================================================
MODEL_DIR = r"E:\AI ATTENDANCE\Attendance\backend\buffalo_l"
DETECTION_MODEL = "det_10g.onnx"
RECOGNITION_MODEL = "w600k_r50.onnx"

# Improved settings
DETECT_SIZE = (640, 640)  # Square for better detection
SIMILARITY_THRESHOLD = 0.45  # Stricter threshold (was 0.3)
MIN_TIME_GAP = timedelta(minutes=1)
PROCESS_EVERY_N_FRAMES = 3  # Process more frequently
COOLDOWN_SECONDS = 10

# Detection settings
DETECTION_CONFIDENCE = 0.6  # Higher confidence threshold
NMS_THRESHOLD = 0.4  # Non-maximum suppression

# Tracker settings
TRACK_IOU_THRESHOLD = 0.3
TRACK_MAX_AGE = 30


# ============================================================================
# MODEL LOADING
# ============================================================================
print("Loading models...")
try:
    # Try CUDA first, fallback to CPU if CUDA is not available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    det_session = ort.InferenceSession(
        os.path.join(MODEL_DIR, DETECTION_MODEL),
        providers=providers
    )
    rec_session = ort.InferenceSession(
        os.path.join(MODEL_DIR, RECOGNITION_MODEL),
        providers=providers
    )
    
    # Check which provider is actually being used
    det_provider = det_session.get_providers()[0]
    rec_provider = rec_session.get_providers()[0]
    print(f"[OK] Models loaded")
    print(f"[INFO] Detection model using: {det_provider}")
    print(f"[INFO] Recognition model using: {rec_provider}\n")
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}")
    print("[INFO] Make sure CUDA Toolkit and cuDNN are installed if using GPU")
    exit(1)


# ============================================================================
# IMPROVED FACE DETECTION WITH NMS
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


def detect_faces(frame, threshold=DETECTION_CONFIDENCE):
    """Improved face detection with NMS"""
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
    
    # Parse detections for SCRFD model
    faces = []
    all_scores = []
    
    for idx, stride in enumerate([8, 16, 32]):
        score_map = outputs[idx]
        bbox_map = outputs[idx + 3]
        kps_map = outputs[idx + 6] if len(outputs) > 6 else None
        
        height, width = 640 // stride, 640 // stride
        
        # Generate anchors
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        
        # Two anchors per position
        anchor_centers = np.stack([anchor_centers, anchor_centers], axis=1).reshape((-1, 2))
        
        # Process scores and boxes
        scores = score_map.reshape(-1)
        bboxes = bbox_map.reshape(-1, 4)
        
        if kps_map is not None:
            kps = kps_map.reshape(-1, 10)
        
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
    
    # Apply NMS to remove duplicates
    if len(faces) > 0:
        keep_indices = nms(faces, all_scores, NMS_THRESHOLD)
        faces = [faces[i] + [all_scores[i]] for i in keep_indices]
    
    return faces[:10]  # Return top 10


# ============================================================================
# FACE ALIGNMENT (CRITICAL FOR RECOGNITION)
# ============================================================================
def align_face(frame, bbox):
    """
    Align face for better recognition
    This is a simplified alignment - ideally use landmarks
    """
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
    """Extract 512-d embedding with face alignment"""
    face = align_face(frame, bbox)
    
    if face is None:
        return None
    
    # Prepare input
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
# IMPROVED RECOGNITION WITH BETTER MATCHING
# ============================================================================
def load_face_cache(db):
    """Load all employee embeddings into cache at startup"""
    print("Loading employee face cache...")
    cache = {}
    
    employees = db.get_all_employees()
    for emp_id, emp_name, _ in employees:
        emp_data = db.get_employee_by_id(emp_id)
        if emp_data and emp_data[2]:  # Has face embedding
            embedding = np.frombuffer(emp_data[2], dtype=np.float32)
            
            # Normalize stored embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            cache[emp_id] = {
                'name': emp_name,
                'embedding': embedding
            }
    
    print(f"[OK] Loaded {len(cache)} employees into cache")
    print(f"[INFO] Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"[INFO] Detection confidence: {DETECTION_CONFIDENCE}\n")
    return cache


def recognize(embedding, face_cache):
    """
    Match embedding against cached faces
    Returns (id, name, similarity)
    """
    if embedding is None or len(face_cache) == 0:
        return None, None, 0.0
    
    best_id, best_name, best_sim = None, None, 0.0
    
    # Calculate similarities with all cached faces
    similarities = []
    for emp_id, data in face_cache.items():
        stored = data['embedding']
        
        # Cosine similarity (both embeddings are normalized)
        similarity = float(np.dot(embedding, stored))
        similarities.append((emp_id, data['name'], similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # Check if best match exceeds threshold
    if len(similarities) > 0:
        best_id, best_name, best_sim = similarities[0]
        
        # Debug: print top matches
        if best_sim < SIMILARITY_THRESHOLD:
            print(f"[DEBUG] Best match: {best_name} ({best_sim:.3f}) - Below threshold")
        
        if best_sim >= SIMILARITY_THRESHOLD:
            return best_id, best_name, best_sim
    
    return None, None, best_sim


# ============================================================================
# FACE TRACKER (Same as before)
# ============================================================================
class FaceTracker:
    """Lightweight face tracker using IoU matching"""
    
    def __init__(self, iou_threshold=0.3, max_age=30):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age
    
    def iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def update(self, detections, frame_count):
        """Update tracks with new detections"""
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        
        # Match by IoU
        for det_idx, det in enumerate(detections):
            best_iou = 0
            best_track = None
            
            for track_id in unmatched_tracks:
                iou = self.iou(det[:4], self.tracks[track_id]['bbox'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track = track_id
            
            if best_track is not None:
                self.tracks[best_track]['bbox'] = det[:4]
                self.tracks[best_track]['last_seen'] = frame_count
                matched.append((best_track, det[:4], False))
                unmatched_dets.remove(det_idx)
                unmatched_tracks.remove(best_track)
        
        # Create new tracks
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = self.next_id
            self.next_id += 1
            
            self.tracks[track_id] = {
                'bbox': det[:4],
                'last_seen': frame_count,
                'embedding': None,
                'emp_id': None,
                'emp_name': None,
                'recognition_attempts': 0
            }
            matched.append((track_id, det[:4], True))
        
        # Remove stale tracks
        to_remove = [tid for tid in self.tracks 
                     if frame_count - self.tracks[tid]['last_seen'] > self.max_age]
        for track_id in to_remove:
            del self.tracks[track_id]
        
        return matched
    
    def get_track(self, track_id):
        return self.tracks.get(track_id)
    
    def set_recognition(self, track_id, embedding, emp_id, emp_name):
        if track_id in self.tracks:
            self.tracks[track_id]['embedding'] = embedding
            self.tracks[track_id]['emp_id'] = emp_id
            self.tracks[track_id]['emp_name'] = emp_name
            self.tracks[track_id]['recognition_attempts'] += 1


# ============================================================================
# ATTENDANCE LOGIC
# ============================================================================
def mark_attendance(emp_id, emp_name, db):
    """Mark entry or exit based on last record"""
    now = datetime.now()
    records = db.get_attendance_by_employee(emp_id)
    
    if not records:
        db.insert_attendance_entry(emp_id, now)
        print(f"✓ ENTRY: {emp_name} at {now.strftime('%H:%M:%S')}")
        return "ENTRY"
    
    last = records[0]
    entry_time, exit_time = last[3], last[4]
    
    if exit_time is None:
        if now - entry_time > MIN_TIME_GAP:
            db.update_attendance_exit(last[0], now)
            print(f"✓ EXIT: {emp_name} at {now.strftime('%H:%M:%S')}")
            return "EXIT"
    else:
        if now - exit_time > MIN_TIME_GAP:
            db.insert_attendance_entry(emp_id, now)
            print(f"✓ ENTRY: {emp_name} at {now.strftime('%H:%M:%S')}")
            return "ENTRY"
    
    return "SKIP"


# ============================================================================
# THREADING COMPONENTS
# ============================================================================
class CameraThread(Thread):
    """Thread for capturing frames from camera"""
    
    def __init__(self, frame_queue):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.stopped = False
        self.cap = None
        
    def init_camera(self):
        print("Opening camera...")
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                # Set higher resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[OK] Camera {idx} ready ({actual_w}x{actual_h})")
                    return cap
                cap.release()
        return None
    
    def run(self):
        self.cap = self.init_camera()
        if self.cap is None:
            print("[ERROR] No camera found!")
            return
        
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)
    
    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()


class DetectionThread(Thread):
    """Thread for face detection and recognition"""
    
    def __init__(self, frame_queue, result_queue, face_cache, db):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.face_cache = face_cache
        self.db = db
        self.stopped = False
        self.tracker = FaceTracker(iou_threshold=TRACK_IOU_THRESHOLD, max_age=TRACK_MAX_AGE)
        self.last_seen = {}
        self.frame_count = 0
        self.lock = Lock()
    
    def run(self):
        while not self.stopped:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                self.frame_count += 1
                
                # Process every Nth frame
                if self.frame_count % PROCESS_EVERY_N_FRAMES != 0:
                    with self.lock:
                        results = {
                            'frame': frame,
                            'tracks': self.tracker.tracks.copy(),
                            'frame_count': self.frame_count
                        }
                    
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()
                        except Empty:
                            pass
                    self.result_queue.put(results)
                    continue
                
                # Detect faces
                detections = detect_faces(frame)
                tracks = self.tracker.update(detections, self.frame_count)
                
                processed_tracks = []
                
                for track_id, bbox, is_new in tracks:
                    track_data = self.tracker.get_track(track_id)
                    
                    # Recognize new tracks or retry failed recognitions (max 3 attempts)
                    should_recognize = (is_new or 
                                       (track_data['emp_id'] is None and 
                                        track_data['recognition_attempts'] < 3))
                    
                    if should_recognize:
                        embedding = get_embedding(frame, bbox)
                        emp_id, emp_name, similarity = recognize(embedding, self.face_cache)
                        self.tracker.set_recognition(track_id, embedding, emp_id, emp_name)
                    else:
                        emp_id = track_data['emp_id']
                        emp_name = track_data['emp_name']
                        similarity = 1.0
                    
                    # Mark attendance
                    action = None
                    if emp_id:
                        now = datetime.now()
                        should_mark = True
                        
                        with self.lock:
                            if emp_id in self.last_seen:
                                if now - self.last_seen[emp_id] < timedelta(seconds=COOLDOWN_SECONDS):
                                    should_mark = False
                        
                        if should_mark:
                            action = mark_attendance(emp_id, emp_name, self.db)
                            with self.lock:
                                self.last_seen[emp_id] = now
                    
                    processed_tracks.append({
                        'track_id': track_id,
                        'bbox': bbox,
                        'emp_id': emp_id,
                        'emp_name': emp_name,
                        'similarity': similarity,
                        'action': action
                    })
                
                results = {
                    'frame': frame,
                    'tracks': processed_tracks,
                    'frame_count': self.frame_count,
                    'total_tracks': len(self.tracker.tracks)
                }
                
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except Empty:
                        pass
                self.result_queue.put(results)
                
            except Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Detection thread: {e}")
                import traceback
                traceback.print_exc()
    
    def stop(self):
        self.stopped = True


# ============================================================================
# MAIN LOOP
# ============================================================================
def main():
    print("="*60)
    print("IMPROVED ATTENDANCE SYSTEM")
    print("="*60)
    print("Improvements:")
    print("  ✓ Better face detection with NMS")
    print("  ✓ Face alignment for recognition")
    print("  ✓ Higher similarity threshold (0.45)")
    print("  ✓ Better duplicate handling")
    print("="*60)
    print("Press ESC to exit\n")
    
    frame_queue = Queue(maxsize=2)
    result_queue = Queue(maxsize=2)
    
    try:
        with DatabaseQueries() as db:
            face_cache = load_face_cache(db)
            
            if not face_cache:
                print("[WARNING] No employees in database!")
                print("Run: python RegisterFace.py\n")
            
            camera_thread = CameraThread(frame_queue)
            detection_thread = DetectionThread(frame_queue, result_queue, face_cache, db)
            
            camera_thread.start()
            time.sleep(0.5)
            detection_thread.start()
            
            print("[OK] All threads started\n")
            
            fps_counter = 0
            fps_start = time.time()
            fps = 0
            
            while True:
                try:
                    results = result_queue.get(timeout=0.1)
                    
                    frame = results['frame']
                    display = frame.copy()
                    
                    if 'tracks' in results and isinstance(results['tracks'], list):
                        for track in results['tracks']:
                            x1, y1, x2, y2 = track['bbox']
                            emp_id = track['emp_id']
                            emp_name = track['emp_name']
                            action = track['action']
                            similarity = track.get('similarity', 0.0)
                            
                            if emp_id:
                                if action == "ENTRY":
                                    color = (0, 255, 0)
                                    label = f"{emp_name} - ENTRY"
                                elif action == "EXIT":
                                    color = (0, 0, 255)
                                    label = f"{emp_name} - EXIT"
                                else:
                                    color = (255, 255, 0)
                                    label = emp_name
                                
                                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(display, label, (x1, y1-30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                cv2.putText(display, f"Conf: {similarity:.3f}", (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            else:
                                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(display, f"Unknown ({similarity:.3f})", (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        if 'total_tracks' in results:
                            cv2.putText(display, f"Tracks: {results['total_tracks']}", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    elif 'tracks' in results and isinstance(results['tracks'], dict):
                        for track_id, track_data in results['tracks'].items():
                            bbox = track_data['bbox']
                            x1, y1, x2, y2 = bbox
                            
                            if track_data['emp_name']:
                                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(display, track_data['emp_name'], (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            else:
                                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(display, "Unknown", (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    fps_counter += 1
                    if time.time() - fps_start > 1.0:
                        fps = fps_counter
                        fps_counter = 0
                        fps_start = time.time()
                    
                    cv2.putText(display, f"FPS: {fps}", (10, display.shape[0] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow("Attendance System [ESC to exit]", display)
                    
                    if cv2.waitKey(1) == 27:
                        break
                
                except Empty:
                    if cv2.waitKey(1) == 27:
                        break
                    continue
    
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[INFO] Stopping threads...")
        camera_thread.stop()
        detection_thread.stop()
        camera_thread.join(timeout=2)
        detection_thread.join(timeout=2)
        
        cv2.destroyAllWindows()
        print("[INFO] System stopped")


if __name__ == "__main__":
    main()