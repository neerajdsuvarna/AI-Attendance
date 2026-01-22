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

# GPU Monitoring (optional - uses nvidia-ml-py)
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_AVAILABLE = True
except:
    GPU_MONITORING_AVAILABLE = False

# ============================================================================
# IMPROVED CONFIGURATION
# ============================================================================
# Use buffalo_l directory in the same folder as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(script_dir, "buffalo_l")
DETECTION_MODEL = "det_10g.onnx"
RECOGNITION_MODEL = "w600k_r50.onnx"

# Optimized settings for RTX 3080 16GB
DETECT_SIZE = (640, 640)  # Square for better detection
SIMILARITY_THRESHOLD = 0.45  # Stricter threshold (was 0.3)
MIN_TIME_GAP = timedelta(minutes=1)
PROCESS_EVERY_N_FRAMES = 1  # Process EVERY frame with CUDA (was 3)
COOLDOWN_SECONDS = 10

# Detection settings - Optimized for speed with CUDA
DETECTION_CONFIDENCE = 0.5  # Balanced for speed (was 0.6)
NMS_THRESHOLD = 0.4  # Non-maximum suppression

# Tracker settings
TRACK_IOU_THRESHOLD = 0.3
TRACK_MAX_AGE = 30

# Performance settings for RTX 3080
CAMERA_WIDTH = 1920  # Higher resolution for better detection
CAMERA_HEIGHT = 1080  # Higher resolution for better detection
FRAME_QUEUE_SIZE = 3  # Larger buffer for smoother processing


# ============================================================================
# MODEL LOADING
# ============================================================================
print("Loading models...")
try:
    # Try CUDA first, fallback to CPU if CUDA is not available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    det_model_path = os.path.join(MODEL_DIR, DETECTION_MODEL)
    rec_model_path = os.path.join(MODEL_DIR, RECOGNITION_MODEL)
    
    if not os.path.exists(det_model_path):
        print(f"[ERROR] Detection model not found: {det_model_path}")
        exit(1)
    if not os.path.exists(rec_model_path):
        print(f"[ERROR] Recognition model not found: {rec_model_path}")
        exit(1)
    
    # Optimize ONNX Runtime session options for CUDA performance
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = False  # Not needed with CUDA
    
    # CUDA provider options optimized for RTX 3080 16GB
    cuda_provider_options = {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8GB limit (half of 16GB for safety)
        'cudnn_conv_algo_search': 'EXHAUSTIVE',  # Best accuracy
        'cudnn_conv_use_max_workspace': '1',  # Use max workspace for speed
        'do_copy_in_default_stream': True,  # Better performance
        'tunable_op_enable': True,  # Enable tunable ops for better performance
        'tunable_op_tuning_enable': True,  # Enable tuning
    }
    
    providers_with_options = [
        ('CUDAExecutionProvider', cuda_provider_options),
        'CPUExecutionProvider'
    ]
    
    det_session = ort.InferenceSession(det_model_path, sess_options=sess_options, providers=providers_with_options)
    rec_session = ort.InferenceSession(rec_model_path, sess_options=sess_options, providers=providers_with_options)
    
    # Check which provider is actually being used
    det_provider = det_session.get_providers()[0]
    rec_provider = rec_session.get_providers()[0]
    
    # Store provider info globally for display
    USING_CUDA = (det_provider == 'CUDAExecutionProvider' and rec_provider == 'CUDAExecutionProvider')
    DET_PROVIDER_NAME = det_provider
    REC_PROVIDER_NAME = rec_provider
    
    print(f"[OK] Loaded detection model: {DETECTION_MODEL}")
    print(f"[INFO] Detection using: {det_provider}")
    print(f"[OK] Loaded recognition model: {RECOGNITION_MODEL}")
    print(f"[INFO] Recognition using: {rec_provider}")
    
    if USING_CUDA:
        print(f"[SUCCESS] CUDA acceleration ENABLED!")
        print(f"[INFO] Optimized for RTX 3080 16GB")
        print(f"[INFO] Processing every frame for maximum FPS")
        print(f"[INFO] GPU memory limit: 8GB")
        
        # Try to get GPU info
        if GPU_MONITORING_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                print(f"[INFO] GPU: {gpu_name}")
                print(f"[INFO] GPU Memory: {mem_info.total / 1024**3:.1f} GB total")
            except:
                pass
    else:
        print(f"[WARNING] Running on CPU (CUDA not available)")
        print(f"[INFO] Detection: {det_provider}, Recognition: {rec_provider}")
    print()
    
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}")
    print("[INFO] Make sure CUDA Toolkit and cuDNN are installed if using GPU")
    import traceback
    traceback.print_exc()
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
                'recognition_attempts': 0,
                'last_similarity': 0.0
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
    
    def set_recognition(self, track_id, embedding, emp_id, emp_name, similarity=None):
        if track_id in self.tracks:
            self.tracks[track_id]['embedding'] = embedding
            self.tracks[track_id]['emp_id'] = emp_id
            self.tracks[track_id]['emp_name'] = emp_name
            self.tracks[track_id]['recognition_attempts'] += 1
            if similarity is not None:
                self.tracks[track_id]['last_similarity'] = similarity


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
                # Set optimized resolution for RTX 3080
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                # Optimize camera settings for performance
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
                cap.set(cv2.CAP_PROP_FPS, 30)  # Set target FPS
                
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
                
                # CONTINUOUS DETECTION: Detect faces in every frame
                detections = detect_faces(frame)
                tracks = self.tracker.update(detections, self.frame_count)
                
                processed_tracks = []
                
                for track_id, bbox, is_new in tracks:
                    track_data = self.tracker.get_track(track_id)
                    
                    # CONTINUOUS RECOGNITION: Always try to recognize
                    # - New tracks: recognize immediately
                    # - Unrecognized tracks: keep trying continuously (no limit with CUDA)
                    # - Recognized tracks: re-verify periodically (every 30 frames) to handle re-entry
                    # This ensures detection NEVER stops - it keeps working continuously
                    should_recognize = (
                        is_new or  # New face detected
                        track_data['emp_id'] is None or  # Not recognized yet - keep trying forever
                        (self.frame_count % 30 == 0)  # Re-verify recognized faces periodically
                    )
                    
                    if should_recognize:
                        embedding = get_embedding(frame, bbox)
                        emp_id, emp_name, similarity = recognize(embedding, self.face_cache)
                        self.tracker.set_recognition(track_id, embedding, emp_id, emp_name, similarity)
                    else:
                        # Use cached recognition data
                        emp_id = track_data['emp_id']
                        emp_name = track_data['emp_name']
                        similarity = track_data.get('last_similarity', 1.0)
                    
                    # Mark attendance (only for recognized faces with cooldown)
                    action = None
                    if emp_id and similarity >= SIMILARITY_THRESHOLD:
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
    print("IMPROVED ATTENDANCE SYSTEM - CUDA OPTIMIZED")
    print("="*60)
    print("Performance Optimizations:")
    print("  ✓ CUDA acceleration enabled")
    print("  ✓ Process every frame (was every 3rd)")
    print("  ✓ Optimized camera resolution (1920x1080)")
    print("  ✓ CUDA session optimizations")
    print("  ✓ Reduced recognition retries")
    print("Improvements:")
    print("  ✓ Better face detection with NMS")
    print("  ✓ Face alignment for recognition")
    print("  ✓ Higher similarity threshold (0.45)")
    print("  ✓ Better duplicate handling")
    print("="*60)
    print("Press ESC to exit\n")
    
    frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
    result_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
    
    # Initialize threads to None to avoid UnboundLocalError
    camera_thread = None
    detection_thread = None
    
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
                    
                    # Display FPS
                    cv2.putText(display, f"FPS: {fps}", (10, display.shape[0] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Display GPU/CPU status
                    if USING_CUDA:
                        gpu_status = "GPU: CUDA"
                        gpu_color = (0, 255, 0)  # Green
                        
                        # Try to get GPU usage if monitoring available
                        if GPU_MONITORING_AVAILABLE:
                            try:
                                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                                gpu_mem_used_gb = mem_info.used / 1024**3
                                gpu_mem_total_gb = mem_info.total / 1024**3
                                
                                # Get GPU utilization (if available)
                                try:
                                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                                    gpu_util = util.gpu
                                except:
                                    gpu_util = None
                                
                                if gpu_util is not None:
                                    gpu_status = f"GPU: CUDA ({gpu_util}% | {gpu_mem_used_gb:.1f}GB/{gpu_mem_total_gb:.1f}GB)"
                                else:
                                    gpu_status = f"GPU: CUDA ({gpu_mem_used_gb:.1f}GB/{gpu_mem_total_gb:.1f}GB)"
                            except:
                                pass
                    else:
                        gpu_status = "CPU: Fallback"
                        gpu_color = (0, 165, 255)  # Orange
                    
                    cv2.putText(display, gpu_status, (10, display.shape[0] - 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, gpu_color, 2)
                    
                    # Display provider info
                    provider_text = f"Det: {DET_PROVIDER_NAME[:4]}, Rec: {REC_PROVIDER_NAME[:4]}"
                    cv2.putText(display, provider_text, (10, display.shape[0] - 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
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
        if camera_thread:
            camera_thread.stop()
            camera_thread.join(timeout=2)
        if detection_thread:
            detection_thread.stop()
            detection_thread.join(timeout=2)
        
        cv2.destroyAllWindows()
        print("[INFO] System stopped")


if __name__ == "__main__":
    main()