"""
AI Attendance System - Backend API
Flask application for handling attendance system backend operations
"""

import os
import sys
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from supabase import create_client, Client
from face_capture import check_face_quality, detect_faces, base64_to_image, process_images_to_embeddings
from live_detection import (
    initialize_detection,
    detect_faces_in_frame,
    base64_to_image as live_base64_to_image,
    load_face_cache_from_edge,
    attendance_tracker
)
import base64
import cv2
import numpy as np

# ─────────────────────────────────────────────────────
# Load environment variables from .env
# ─────────────────────────────────────────────────────

# Load environment variables from backend .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ─────────────────────────────────────────────────────
# Global paths and configuration
# ─────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ─────────────────────────────────────────────────────
# Supabase Configuration
# ─────────────────────────────────────────────────────

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Validate required environment variables
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL environment variable is required")
if not SUPABASE_ANON_KEY:
    raise ValueError("SUPABASE_ANON_KEY environment variable is required")
if not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable is required")

# Initialize Supabase clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Initialize live detection system
try:
    initialize_detection(SUPABASE_URL, SUPABASE_ANON_KEY)
    print("[OK] Live detection system initialized")
except Exception as e:
    print(f"[WARNING] Failed to initialize live detection: {e}")

# ─────────────────────────────────────────────────────
# Flask app configuration
# ─────────────────────────────────────────────────────

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure CORS for development
CORS(app,
     supports_credentials=True,
     origins=[
         "http://localhost:3000",  # Frontend dev server (Vite default)
         "http://localhost:5173",  # Alternative Vite port
         "http://127.0.0.1:3000",  # Alternative localhost
         "http://127.0.0.1:5173",  # Alternative localhost
     ],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept"])

# Initialize Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─────────────────────────────────────────────────────
# Authentication Helper Functions
# ─────────────────────────────────────────────────────

def get_auth_token():
    """Extract Bearer token from Authorization header"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    return auth_header.split(' ')[1]

def verify_supabase_token():
    """Verify Supabase JWT token and return user data"""
    token = get_auth_token()
    if not token:
        return None, {"error": "No valid authorization header"}, 401
    
    try:
        # Verify token with Supabase
        user_response = supabase.auth.get_user(token)
        if user_response.user:
            return user_response.user, None, None
        else:
            return None, {"error": "Invalid token"}, 401
    except Exception as e:
        print(f"Token verification error: {e}")
        return None, {"error": "Token verification failed"}, 401

# ─────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint - no authentication required"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AI Attendance System API"
    }), 200

@app.route('/api/test', methods=['GET'])
def test_api():
    """Test API endpoint - requires Supabase authentication"""
    user, error_response, status_code = verify_supabase_token()
    
    if error_response:
        return jsonify(error_response), status_code
    
    try:
        return jsonify({
            "message": "Authentication successful!",
            "user": {
                "id": user.id,
                "email": user.email,
                "created_at": user.created_at
            },
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        print(f"Error in test API: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/face/check-quality', methods=['POST'])
def check_face_quality_api():
    """
    Check if a single captured image has good face quality
    Requires authentication
    """
    user, error_response, status_code = verify_supabase_token()
    
    if error_response:
        return jsonify(error_response), status_code
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Missing 'image' field in request"}), 400
        
        image_base64 = data['image']
        
        # Convert base64 to OpenCV image
        frame = base64_to_image(image_base64)
        if frame is None:
            return jsonify({
                "success": False,
                "quality_ok": False,
                "message": "Failed to decode image"
            }), 400
        
        # Detect faces
        faces = detect_faces(frame, threshold=0.5)
        
        if len(faces) == 0:
            return jsonify({
                "success": True,
                "quality_ok": False,
                "message": "No face detected",
                "has_face": False
            }), 200
        
        # Get best face (highest confidence)
        faces.sort(key=lambda x: x[4], reverse=True)
        best_face = faces[0]
        
        # Check quality
        quality_ok, quality_msg = check_face_quality(frame, best_face)
        
        return jsonify({
            "success": True,
            "quality_ok": quality_ok,
            "message": quality_msg,
            "has_face": True,
            "confidence": float(best_face[4]),
            "face_count": len(faces)
        }), 200
        
    except Exception as e:
        print(f"Error in check_face_quality_api: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/api/face/process-embeddings', methods=['POST'])
def process_face_embeddings_api():
    """
    Process multiple face images and return embeddings
    Requires authentication
    """
    user, error_response, status_code = verify_supabase_token()
    
    if error_response:
        return jsonify(error_response), status_code
    
    try:
        data = request.get_json()
        if not data or 'images' not in data:
            return jsonify({"error": "Missing 'images' field in request"}), 400
        
        images = data['images']
        if not isinstance(images, list) or len(images) == 0:
            return jsonify({"error": "'images' must be a non-empty list"}), 400
        
        # Process images to embeddings
        result = process_images_to_embeddings(images)
        
        if result['success']:
            return jsonify({
                "success": True,
                "message": result['message'],
                "embeddings": result['embeddings'],
                "average_embedding": result['average_embedding'],
                "angles_captured": result['angles_captured']
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": result['message'],
                "embeddings": result.get('embeddings', []),
                "average_embedding": result.get('average_embedding')
            }), 400
        
    except Exception as e:
        print(f"Error in process_face_embeddings_api: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/api/face/live-detect', methods=['POST'])
def live_detect_api():
    """
    Live face detection endpoint
    Takes a video frame (base64 image) and returns detected/recognized faces
    Requires authentication
    """
    user, error_response, status_code = verify_supabase_token()
    
    if error_response:
        return jsonify(error_response), status_code
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Missing 'image' field in request"}), 400
        
        image_base64 = data['image']
        
        # Get auth token for edge function
        auth_token = get_auth_token()
        if not auth_token:
            return jsonify({"error": "Missing authorization token"}), 401
        
        # Convert base64 to OpenCV image
        frame = live_base64_to_image(image_base64)
        if frame is None:
            return jsonify({
                "success": False,
                "error": "Failed to decode image"
            }), 400
        
        # Detect and recognize faces
        detections = detect_faces_in_frame(frame, auth_token)
        
        # Format response
        detected_employees = []
        for det in detections:
            if det['recognized']:
                detected_employees.append({
                    "employee_id": det['employee_id'],
                    "name": det['employee_name'],
                    "similarity": det['similarity'],
                    "bbox": det['bbox'],
                    "confidence": det['confidence']
                })
        
        return jsonify({
            "success": True,
            "detected": len(detected_employees) > 0,
            "face_count": len(detections),
            "recognized_count": len(detected_employees),
            "detections": detections,
            "employees": detected_employees,
            "best_match": detected_employees[0] if detected_employees else None
        }), 200
        
    except Exception as e:
        print(f"Error in live_detect_api: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/api/face/reload-cache', methods=['POST'])
def reload_face_cache_api():
    """
    Reload employee embeddings cache from edge function
    Requires authentication
    """
    user, error_response, status_code = verify_supabase_token()
    
    if error_response:
        return jsonify(error_response), status_code
    
    try:
        auth_token = get_auth_token()
        if not auth_token:
            return jsonify({"error": "Missing authorization token"}), 401
        
        # Reload cache
        cache = load_face_cache_from_edge(auth_token)
        
        return jsonify({
            "success": True,
            "message": f"Cache reloaded successfully",
            "employee_count": len(cache)
        }), 200
        
    except Exception as e:
        print(f"Error in reload_face_cache_api: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# ─────────────────────────────────────────────────────
# Socket.IO Handlers for Live Detection
# ─────────────────────────────────────────────────────

def decode_image_base64(img_data):
    """Decode base64 image string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERROR] Failed to decode base64 image: {e}")
        return None

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected to face detection socket')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected from face detection socket')

@socketio.on('detection_frame')
def handle_detection_frame(data):
    """Handle incoming video frame for face detection"""
    try:
        img_data = data.get("image")
        
        if not img_data:
            emit("detection_response", {"error": "No image data provided"})
            return
        
        # Decode image
        frame = decode_image_base64(img_data)
        if frame is None:
            print("[ERROR] Failed to decode image data")
            emit("detection_response", {"error": "Invalid image data"})
            return
        
        # Get auth token from frame data
        auth_token = data.get("auth_token")
        
        # Detect faces in frame (with attendance tracking enabled)
        detections = detect_faces_in_frame(frame, auth_token, track_attendance=True)
        
        # Format response
        detected_employees = []
        attendance_actions = []
        for det in detections:
            if det['recognized']:
                detected_employees.append({
                    "employee_id": det['employee_id'],
                    "name": det['employee_name'],
                    "similarity": det['similarity'],
                    "bbox": det['bbox'],
                    "confidence": det['confidence']
                })
                # Include attendance action if present
                if det.get('attendance_action'):
                    attendance_actions.append({
                        "employee_id": det['employee_id'],
                        "employee_name": det['employee_name'],
                        "action": det['attendance_action']
                    })
        
        emit("detection_response", {
            "success": True,
            "detected": len(detected_employees) > 0,
            "face_count": len(detections),
            "recognized_count": len(detected_employees),
            "detections": detections,
            "employees": detected_employees,
            "best_match": detected_employees[0] if detected_employees else None,
            "attendance_actions": attendance_actions  # New field for attendance updates
        })
        
    except Exception as e:
        print(f"[ERROR] Exception in handle_detection_frame: {e}")
        import traceback
        traceback.print_exc()
        emit("detection_response", {"error": f"Internal server error: {str(e)}"})

@socketio.on('stop_detection')
def handle_stop_detection():
    """Handle detection stop - reset attendance tracker"""
    try:
        if attendance_tracker:
            attendance_tracker.reset()
            print("[INFO] Attendance tracker reset")
        emit("detection_stopped", {"success": True})
    except Exception as e:
        print(f"[ERROR] Exception in handle_stop_detection: {e}")
        emit("detection_stopped", {"error": str(e)})

# ─────────────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large"}), 413

# ─────────────────────────────────────────────────────
# Application Entry Point
# ─────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("AI Attendance System - Backend API")
    print("=" * 60)
    print(f"Supabase URL: {SUPABASE_URL}")
    print(f"Environment: {'Development' if app.debug else 'Production'}")
    print("=" * 60)
    
    # Run Flask development server with Socket.IO
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=True,
        allow_unsafe_werkzeug=True
    )

