"""
Live Face Detection Module
Uses edge function to fetch employee embeddings and performs real-time detection
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
import base64
import requests
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import threading

# ============================================================================
# CONFIGURATION
# ============================================================================
# Get the directory where this script is located, then look for buffalo_l folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "buffalo_l")
DETECTION_MODEL = "det_10g.onnx"
RECOGNITION_MODEL = "w600k_r50.onnx"

# Detection settings
DETECT_SIZE = (640, 640)
SIMILARITY_THRESHOLD = 0.45
DETECTION_CONFIDENCE = 0.6
NMS_THRESHOLD = 0.4

# Edge function URL (will be set from environment or app config)
EDGE_FUNCTION_URL = None
SUPABASE_URL = None
SUPABASE_ANON_KEY = None

# Global model sessions
det_session = None
rec_session = None
face_cache = {}  # Cache for employee embeddings
cache_lock = threading.Lock()  # Lock for thread-safe cache operations
cache_loading = False  # Flag to prevent concurrent cache loads

# Attendance tracking configuration
ATTENDANCE_CONFIG = {
    'MIN_CONSECUTIVE_DETECTIONS': 5,  # Number of consecutive frames to mark entry (5 frames = ~1 second at 5 FPS)
    'EXIT_STRATEGY': 'hybrid',  # Options: 'timeout', 'frame_based', 'hybrid'
    'EXIT_FRAME_COUNT': 15,  # Consecutive frames of absence to mark exit (frame-based strategy)
    'EXIT_TIMEOUT_SECONDS': 30,  # Seconds of absence before marking exit (timeout strategy / fallback)
    'EXIT_ZONE_THRESHOLD': 0.1,  # Percentage of frame edge considered exit zone (0.1 = 10%)
    'EXIT_MOVEMENT_FRAMES': 3,  # Frames to track movement toward exit zone
    'COOLDOWN_SECONDS': 5  # Cooldown between attendance actions (prevents duplicate entries)
}


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_models():
    """Load ONNX models for detection and recognition with optimized CPU settings"""
    global det_session, rec_session
    
    if det_session is None or rec_session is None:
        print("Loading face recognition models with CPU optimization...")
        try:
            # Configure session options for better CPU performance
            sess_options = ort.SessionOptions()
            
            # Use all available CPU threads (0 = use all)
            sess_options.intra_op_num_threads = 0  # Use all cores for single operations
            sess_options.inter_op_num_threads = 0  # Use all cores for parallel operations
            
            # Enable optimizations
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Set execution mode for better performance
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # CPU provider options for better performance
            cpu_provider_options = {
                'arena_extend_strategy': 'kSameAsRequested',
                'enable_cpu_mem_arena': True,
            }
            
            # Try to use GPU if available, fallback to optimized CPU
            available_providers = ort.get_available_providers()
            providers = []
            
            if 'CUDAExecutionProvider' in available_providers:
                print("[INFO] CUDA GPU available - using GPU acceleration")
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                print(f"[INFO] Using CPU with {os.cpu_count()} cores")
                providers = ['CPUExecutionProvider']
            
            det_session = ort.InferenceSession(
                os.path.join(MODEL_DIR, DETECTION_MODEL),
                sess_options=sess_options,
                providers=providers,
                provider_options=[cpu_provider_options] if 'CUDAExecutionProvider' not in providers else []
            )
            
            rec_session = ort.InferenceSession(
                os.path.join(MODEL_DIR, RECOGNITION_MODEL),
                sess_options=sess_options,
                providers=providers,
                provider_options=[cpu_provider_options] if 'CUDAExecutionProvider' not in providers else []
            )
            
            print(f"[OK] Models loaded successfully using: {det_session.get_providers()}")
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            raise
    return det_session, rec_session


# ============================================================================
# FACE DETECTION
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
    """Detect faces in frame using ONNX model"""
    if det_session is None:
        load_models()
    
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
        kps_map = outputs[idx + 6] if len(outputs) > 6 else None
        
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
        keep_indices = nms(faces, all_scores, NMS_THRESHOLD)
        faces = [faces[i] + [all_scores[i]] for i in keep_indices]
    
    return faces[:10]  # Return top 10


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
    
    if face.shape[0] < 40 or face.shape[1] < 40:
        return None
    
    return face


def get_embedding(frame, bbox):
    """Extract 512-d embedding from face"""
    if rec_session is None:
        load_models()
    
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
# FETCH EMPLOYEE EMBEDDINGS FROM EDGE FUNCTION
# ============================================================================
def fetch_employee_embeddings(auth_token: str) -> Dict:
    """
    Fetch employee embeddings from Supabase edge function
    Returns: Dict with 'success', 'employees' list
    """
    global face_cache, EDGE_FUNCTION_URL, SUPABASE_URL
    
    if not EDGE_FUNCTION_URL:
        # Construct edge function URL from Supabase URL
        if not SUPABASE_URL:
            raise ValueError("SUPABASE_URL not set. Call initialize_detection() first.")
        EDGE_FUNCTION_URL = f"{SUPABASE_URL}/functions/v1/get-employee-embeddings"
    
    try:
        response = requests.post(
            EDGE_FUNCTION_URL,
            headers={
                'Authorization': f'Bearer {auth_token}',
                'Content-Type': 'application/json'
            },
            json={},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return data
            else:
                return {'success': False, 'error': data.get('error', 'Unknown error')}
        else:
            return {
                'success': False,
                'error': f'HTTP {response.status_code}: {response.text}'
            }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def load_face_cache_from_edge(auth_token: str) -> Dict:
    """
    Load employee embeddings from edge function and build cache (thread-safe)
    Returns: face_cache dict with employee_id -> {name, embedding}
    """
    global face_cache, cache_lock, cache_loading
    
    # Check if cache is already loaded (fast path, no lock needed)
    if len(face_cache) > 0:
        return face_cache
    
    # Acquire lock to prevent concurrent loading
    with cache_lock:
        # Double-check after acquiring lock (another thread might have loaded it)
        if len(face_cache) > 0:
            return face_cache
        
        # Check if another thread is already loading
        if cache_loading:
            # Wait a bit and check again
            import time
            time.sleep(0.1)
            if len(face_cache) > 0:
                return face_cache
        
        # Set loading flag
        cache_loading = True
        
        try:
            result = fetch_employee_embeddings(auth_token)
            
            if not result.get('success'):
                print(f"[ERROR] Failed to fetch embeddings: {result.get('error')}")
                return {}
            
            employees = result.get('employees', [])
            cache = {}
            
            for emp in employees:
                emp_id = emp['id']
                emp_name = emp['name']
                embeddings_b64 = emp.get('face_embeddings')
                
                if embeddings_b64:
                    try:
                        # Decode base64 embedding
                        embedding_bytes = base64.b64decode(embeddings_b64)
                        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                        
                        # Normalize
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
                        
                        cache[emp_id] = {
                            'name': emp_name,
                            'email': emp.get('email', ''),
                            'embedding': embedding
                        }
                    except Exception as e:
                        print(f"[WARNING] Failed to decode embedding for {emp_name}: {e}")
                        continue
            
            face_cache = cache
            print(f"[OK] Loaded {len(cache)} employee embeddings into cache")
            return cache
        finally:
            # Always clear loading flag
            cache_loading = False


# ============================================================================
# FACE RECOGNITION
# ============================================================================
def recognize_face(embedding, face_cache: Dict) -> Tuple[Optional[str], Optional[str], float]:
    """
    Match embedding against cached faces
    Returns: (employee_id, employee_name, similarity)
    """
    if embedding is None or len(face_cache) == 0:
        return None, None, 0.0
    
    best_id, best_name, best_sim = None, None, 0.0
    
    # Calculate similarities
    for emp_id, data in face_cache.items():
        stored = data['embedding']
        similarity = float(np.dot(embedding, stored))
        
        if similarity > best_sim:
            best_sim = similarity
            best_id = emp_id
            best_name = data['name']
    
    # Check if best match exceeds threshold
    if best_sim >= SIMILARITY_THRESHOLD:
        return best_id, best_name, best_sim
    
    return None, None, best_sim


# ============================================================================
# MAIN DETECTION FUNCTION
# ============================================================================
def detect_faces_in_frame(frame: np.ndarray, auth_token: str = None, track_attendance: bool = True) -> List[Dict]:
    """
    Detect and recognize faces in a single frame
    Args:
        frame: OpenCV BGR image (numpy array)
        auth_token: Supabase auth token (optional if cache already loaded)
        track_attendance: Whether to track and mark attendance
    Returns:
        List of detections with bbox, employee_id, name, similarity, attendance_action
    """
    global face_cache, attendance_tracker
    
    # Load models if not loaded
    load_models()
    
    # Load face cache if empty and token provided (thread-safe)
    if len(face_cache) == 0 and auth_token:
        load_face_cache_from_edge(auth_token)
    
    # Update attendance tracker auth token if provided
    if attendance_tracker and auth_token:
        attendance_tracker.update_auth_token(auth_token)
    
    # Detect faces
    detections = detect_faces(frame)
    
    results = []
    
    for det in detections:
        bbox = det[:4]
        confidence = det[4] if len(det) > 4 else 1.0
        
        # Extract embedding
        embedding = get_embedding(frame, bbox)
        
        if embedding is not None:
            # Recognize
            emp_id, emp_name, similarity = recognize_face(embedding, face_cache)
            
            results.append({
                'bbox': bbox,
                'confidence': float(confidence),
                'employee_id': emp_id,
                'employee_name': emp_name,
                'similarity': float(similarity),
                'recognized': emp_id is not None,
                'attendance_action': None
            })
        else:
            results.append({
                'bbox': bbox,
                'confidence': float(confidence),
                'employee_id': None,
                'employee_name': None,
                'similarity': 0.0,
                'recognized': False,
                'attendance_action': None
            })
    
    # Process attendance tracking if enabled
    if track_attendance and attendance_tracker:
        # Get frame dimensions for exit zone calculation
        frame_height, frame_width = frame.shape[:2]
        results = attendance_tracker.process_detections(results, frame_width, frame_height)
    
    return results


# ============================================================================
# ATTENDANCE TRACKING
# ============================================================================
class AttendanceTracker:
    """
    Tracks employee presence and marks attendance in the database
    """
    def __init__(self, supabase_url: str, auth_token: str = None):
        self.supabase_url = supabase_url
        self.auth_token = auth_token
        self.edge_function_url = f"{supabase_url}/functions/v1/mark-attendance"
        
        # Track employee presence state
        # Format: {employee_id: {
        #   'consecutive_detections': int,
        #   'consecutive_absences': int,  # Frames not detected
        #   'last_seen': datetime,
        #   'last_bbox': [x1, y1, x2, y2],  # Last known position
        #   'bbox_history': [(x1, y1, x2, y2), ...],  # Recent bbox positions for movement tracking
        #   'moving_toward_exit': bool,  # Whether bbox was moving toward exit zone
        #   'marked_entry': bool,
        #   'last_action_time': datetime
        # }}
        self.employee_states = {}
        self.lock = threading.Lock()
        
        # Configuration
        self.min_consecutive = ATTENDANCE_CONFIG['MIN_CONSECUTIVE_DETECTIONS']
        self.exit_strategy = ATTENDANCE_CONFIG.get('EXIT_STRATEGY', 'hybrid')
        self.exit_frame_count = ATTENDANCE_CONFIG.get('EXIT_FRAME_COUNT', 15)
        self.exit_timeout = timedelta(seconds=ATTENDANCE_CONFIG['EXIT_TIMEOUT_SECONDS'])
        self.exit_zone_threshold = ATTENDANCE_CONFIG.get('EXIT_ZONE_THRESHOLD', 0.1)
        self.exit_movement_frames = ATTENDANCE_CONFIG.get('EXIT_MOVEMENT_FRAMES', 3)
        self.cooldown = timedelta(seconds=ATTENDANCE_CONFIG['COOLDOWN_SECONDS'])
    
    def update_auth_token(self, auth_token: str):
        """Update the auth token for API calls"""
        self.auth_token = auth_token
    
    def mark_attendance(self, employee_id: str, action: str) -> Dict:
        """
        Call edge function to mark attendance
        Returns: {'success': bool, 'message': str, ...}
        """
        if not self.auth_token:
            return {'success': False, 'error': 'No auth token provided'}
        
        try:
            response = requests.post(
                self.edge_function_url,
                headers={
                    'Authorization': f'Bearer {self.auth_token}',
                    'Content-Type': 'application/json'
                },
                json={
                    'employee_id': employee_id,
                    'action': action
                },
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}: {response.text}'
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _is_in_exit_zone(self, bbox: List[int], frame_width: int, frame_height: int) -> bool:
        """Check if bounding box is in exit zone (edges of frame)"""
        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        # Check if center is near any edge
        left_threshold = frame_width * self.exit_zone_threshold
        right_threshold = frame_width * (1 - self.exit_zone_threshold)
        top_threshold = frame_height * self.exit_zone_threshold
        
        return (bbox_center_x < left_threshold or 
                bbox_center_x > right_threshold or 
                bbox_center_y < top_threshold)
    
    def _is_moving_toward_exit(self, bbox_history: List[List[int]], frame_width: int, frame_height: int) -> bool:
        """Check if bounding box was moving toward exit zone"""
        if len(bbox_history) < 2:
            return False
        
        # Check if recent bboxes show movement toward edges
        recent_bboxes = bbox_history[-self.exit_movement_frames:]
        exit_zone_count = sum(1 for bbox in recent_bboxes 
                             if self._is_in_exit_zone(bbox, frame_width, frame_height))
        
        # If most recent positions were in exit zone, person was moving toward exit
        return exit_zone_count >= len(recent_bboxes) * 0.5
    
    def _should_mark_exit(self, state: Dict, now: datetime, frame_width: int = 1920, frame_height: int = 1080) -> bool:
        """
        Determine if exit should be marked based on selected strategy
        Returns: (should_mark, reason)
        """
        if not state['marked_entry']:
            return False, None
        
        time_since_last_seen = now - state['last_seen']
        
        # Check cooldown first
        if state['last_action_time']:
            time_since_last_action = now - state['last_action_time']
            if time_since_last_action < self.cooldown:
                return False, "cooldown"
        
        # Strategy 1: Frame-based exit (consistent with entry logic)
        if self.exit_strategy == 'frame_based':
            if state['consecutive_absences'] >= self.exit_frame_count:
                return True, "frame_based"
        
        # Strategy 2: Timeout-based exit
        elif self.exit_strategy == 'timeout':
            if time_since_last_seen >= self.exit_timeout:
                return True, "timeout"
        
        # Strategy 3: Hybrid (recommended)
        elif self.exit_strategy == 'hybrid':
            # Check if person was moving toward exit zone
            moving_toward_exit = False
            if state.get('bbox_history') and len(state['bbox_history']) > 0:
                moving_toward_exit = self._is_moving_toward_exit(
                    state['bbox_history'], frame_width, frame_height
                )
            
            # If moving toward exit, use faster frame-based threshold
            if moving_toward_exit:
                # Faster exit if person was clearly leaving
                fast_exit_frames = max(5, self.exit_frame_count // 2)  # Half the normal threshold
                if state['consecutive_absences'] >= fast_exit_frames:
                    return True, "hybrid_movement"
            
            # Normal frame-based exit
            if state['consecutive_absences'] >= self.exit_frame_count:
                return True, "hybrid_frame"
            
            # Fallback timeout (safety net)
            if time_since_last_seen >= self.exit_timeout:
                return True, "hybrid_timeout"
        
        return False, None
    
    def process_detections(self, detections: List[Dict], frame_width: int = 1920, frame_height: int = 1080) -> List[Dict]:
        """
        Process detections and mark attendance when thresholds are met
        Args:
            detections: List of detection dictionaries
            frame_width: Video frame width (for exit zone calculation)
            frame_height: Video frame height (for exit zone calculation)
        Returns: detections with attendance_action field added
        """
        now = datetime.now()
        recognized_employee_ids = set()
        
        with self.lock:
            # Process each recognized employee
            for det in detections:
                if not det.get('recognized') or not det.get('employee_id'):
                    continue
                
                emp_id = det['employee_id']
                recognized_employee_ids.add(emp_id)
                bbox = det.get('bbox', [])
                
                # Initialize state if needed
                if emp_id not in self.employee_states:
                    self.employee_states[emp_id] = {
                        'consecutive_detections': 0,
                        'consecutive_absences': 0,
                        'last_seen': now,
                        'last_bbox': bbox,
                        'bbox_history': [],
                        'moving_toward_exit': False,
                        'marked_entry': False,
                        'last_action_time': None
                    }
                
                state = self.employee_states[emp_id]
                
                # Reset absence counter (employee is detected)
                state['consecutive_absences'] = 0
                
                # Update consecutive detections
                state['consecutive_detections'] += 1
                state['last_seen'] = now
                
                # Update bbox tracking
                if len(bbox) >= 4:
                    state['last_bbox'] = bbox
                    # Keep recent bbox history for movement tracking
                    state['bbox_history'].append(bbox)
                    # Keep only last N bboxes
                    max_history = self.exit_movement_frames * 2
                    if len(state['bbox_history']) > max_history:
                        state['bbox_history'] = state['bbox_history'][-max_history:]
                    
                    # Check if currently in exit zone
                    state['moving_toward_exit'] = self._is_in_exit_zone(bbox, frame_width, frame_height)
                
                # Check if we should mark entry
                if (not state['marked_entry'] and 
                    state['consecutive_detections'] >= self.min_consecutive):
                    
                    # Check cooldown
                    can_mark = True
                    if state['last_action_time']:
                        time_since_last = now - state['last_action_time']
                        if time_since_last < self.cooldown:
                            can_mark = False
                    
                    if can_mark:
                        result = self.mark_attendance(emp_id, 'entry')
                        if result.get('success'):
                            state['marked_entry'] = True
                            state['last_action_time'] = now
                            det['attendance_action'] = 'entry'
                            print(f"[ATTENDANCE] Entry marked for {det.get('employee_name', emp_id)}")
                        else:
                            print(f"[ATTENDANCE ERROR] Failed to mark entry: {result.get('error')}")
                            det['attendance_action'] = None
                    else:
                        det['attendance_action'] = None
                else:
                    det['attendance_action'] = None
            
            # Check for exits (employees not detected in this frame)
            employees_to_check = list(self.employee_states.keys())
            for emp_id in employees_to_check:
                if emp_id not in recognized_employee_ids:
                    state = self.employee_states[emp_id]
                    
                    # Reset consecutive detections
                    state['consecutive_detections'] = 0
                    
                    # Increment absence counter
                    state['consecutive_absences'] = state.get('consecutive_absences', 0) + 1
                    
                    # Check if we should mark exit using selected strategy
                    should_mark, reason = self._should_mark_exit(state, now, frame_width, frame_height)
                    
                    if should_mark:
                        result = self.mark_attendance(emp_id, 'exit')
                        if result.get('success'):
                            state['marked_entry'] = False
                            state['last_action_time'] = now
                            state['consecutive_absences'] = 0
                            state['bbox_history'] = []  # Clear history
                            print(f"[ATTENDANCE] Exit marked for employee {emp_id} (reason: {reason})")
                        else:
                            print(f"[ATTENDANCE ERROR] Failed to mark exit: {result.get('error')}")
        
        return detections
    
    def reset(self):
        """Reset all tracking state (e.g., when detection stops)"""
        with self.lock:
            self.employee_states.clear()


# Global attendance tracker instance
attendance_tracker = None


def initialize_detection(supabase_url: str, supabase_anon_key: str = None):
    """Initialize detection system with Supabase configuration"""
    global SUPABASE_URL, SUPABASE_ANON_KEY, EDGE_FUNCTION_URL, attendance_tracker
    
    SUPABASE_URL = supabase_url
    if supabase_anon_key:
        SUPABASE_ANON_KEY = supabase_anon_key
    
    # Construct edge function URL
    EDGE_FUNCTION_URL = f"{SUPABASE_URL}/functions/v1/get-employee-embeddings"
    
    # Initialize attendance tracker
    attendance_tracker = AttendanceTracker(supabase_url)
    
    # Load models
    load_models()
    
    print("[OK] Detection system initialized")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def base64_to_image(base64_string: str) -> Optional[np.ndarray]:
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERROR] Failed to decode base64 image: {e}")
        return None

