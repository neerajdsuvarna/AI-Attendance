"""
Simple test script for face_capture.py
Tests the process_images_to_embeddings function with 5 images
"""

import os
import base64
from face_capture import process_images_to_embeddings

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    try:
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            # Add data URL prefix (same format as frontend sends)
            return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        print(f"[ERROR] Failed to read image {image_path}: {e}")
        return None

def test_face_capture(image_paths):
    """
    Test face capture with provided image paths
    
    Args:
        image_paths: List of tuples [(angle_name, path), ...]
                    Example: [("Front", "path/to/front.jpg"), ...]
    """
    
    print("=" * 70)
    print("FACE CAPTURE TEST")
    print("=" * 70)
    print()
    
    # Convert images to base64
    images_data = []
    for angle, path in image_paths:
        print(f"[INFO] Loading {angle} image: {path}")
        
        if not os.path.exists(path):
            print(f"  [ERROR] File not found: {path}")
            continue
        
        image_base64 = image_to_base64(path)
        
        if image_base64:
            images_data.append({
                'angle': angle,
                'image': image_base64
            })
            print(f"  [OK] Loaded successfully")
        else:
            print(f"  [ERROR] Failed to load")
    
    if len(images_data) == 0:
        print("\n[ERROR] No images could be loaded!")
        return None
    
    print(f"\n[INFO] Successfully loaded {len(images_data)} images")
    print("\n[INFO] Processing images and extracting embeddings...")
    print()
    
    # Call the function
    result = process_images_to_embeddings(images_data)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    
    if result['success']:
        print(f"[SUCCESS] ✓ Processing completed!")
        print(f"  Angles processed: {result['angles_captured']}")
        print(f"  Message: {result['message']}")
        print()
        
        print("Individual Embeddings:")
        for i, emb_data in enumerate(result['embeddings'], 1):
            print(f"  {i}. {emb_data['angle']}:")
            print(f"     - Confidence: {emb_data['confidence']:.4f}")
            print(f"     - Embedding (base64): {emb_data['embedding'][:50]}...")
        
        print()
        print("Average Embedding:")
        print(f"  - Base64 length: {len(result['average_embedding'])} chars")
        print(f"  - First 50 chars: {result['average_embedding'][:50]}...")
        
        # Decode and show embedding stats
        try:
            import numpy as np
            avg_emb_bytes = base64.b64decode(result['average_embedding'])
            avg_emb = np.frombuffer(avg_emb_bytes, dtype=np.float32)
            print(f"  - Shape: {avg_emb.shape}")
            print(f"  - Norm: {np.linalg.norm(avg_emb):.4f}")
            print(f"  - Min: {avg_emb.min():.4f}, Max: {avg_emb.max():.4f}")
        except Exception as e:
            print(f"  - Could not decode embedding: {e}")
        
        print("\n[OK] Embeddings are ready for storage!")
        
    else:
        print(f"[FAILED] ✗ Processing failed!")
        print(f"  Error: {result['message']}")
        if result.get('embeddings'):
            print(f"  Partial results: {len(result['embeddings'])} angles processed")
    
    print("\n" + "=" * 70)
    
    return result


if __name__ == "__main__":
    # ========================================================================
    # UPDATE THESE PATHS WITH YOUR IMAGE FILES
    # ========================================================================
    image_paths = [
        ("Front", r"C:\Users\neera\OneDrive\Pictures\Camera Roll 1\WIN_20260108_10_30_47_Pro.jpg"),
        ("Left", r"C:\Users\neera\OneDrive\Pictures\Camera Roll 1\WIN_20260108_10_32_37_Pro.jpg"),
        ("Right", r"C:\Users\neera\OneDrive\Pictures\Camera Roll 1\WIN_20260108_10_32_24_Pro.jpg"),
        ("Up", r"C:\Users\neera\OneDrive\Pictures\Camera Roll 1\WIN_20260108_10_30_55_Pro.jpg"),
        ("Down", r"C:\Users\neera\OneDrive\Pictures\Camera Roll 1\WIN_20260108_10_30_57_Pro.jpg"),
    ]
    
    # Alternative: Use command line arguments
    import sys
    if len(sys.argv) > 1:
        # Usage: python test_face_capture.py front.jpg left.jpg right.jpg up.jpg down.jpg
        if len(sys.argv) == 6:
            image_paths = [
                ("Front", sys.argv[1]),
                ("Left", sys.argv[2]),
                ("Right", sys.argv[3]),
                ("Up", sys.argv[4]),
                ("Down", sys.argv[5]),
            ]
        else:
            print("Usage: python test_face_capture.py <front.jpg> <left.jpg> <right.jpg> <up.jpg> <down.jpg>")
            print("Or update the image_paths list in the script")
            sys.exit(1)
    
    # Run the test
    result = test_face_capture(image_paths)
    
    if result and result['success']:
        print("\n✓ Test passed! Embeddings extracted successfully.")
    else:
        print("\n✗ Test failed. Check the errors above.")
