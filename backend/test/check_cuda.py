"""
CUDA Diagnostic Script
Run this to check if CUDA DLLs are properly installed
"""
import os

print("=" * 60)
print("CUDA Installation Diagnostic")
print("=" * 60)

cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
cuda_versions = ['v12.6', 'v12.5', 'v12.4', 'v12.3', 'v12.2', 'v12.1', 'v12.0', 'v11.8']

print("\n1. Checking CUDA Installation...")
cuda_found = False
cuda_bin = None

for version in cuda_versions:
    test_path = os.path.join(cuda_base, version, 'bin')
    if os.path.exists(test_path):
        cuda_bin = test_path
        print(f"   ✓ Found CUDA {version} at: {test_path}")
        cuda_found = True
        break

if not cuda_found:
    print("   ✗ CUDA Toolkit not found!")
    print("   Please install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads")
    exit(1)

print("\n2. Checking Required DLLs...")
required_dlls = {
    'cublas64_12.dll': 'CUDA Basic Linear Algebra Subroutines (Required)',
    'cublasLt64_12.dll': 'CUDA BLAS Light (Required)',
    'cudnn64_9.dll': 'cuDNN 9.x (Required for ONNX Runtime)',
    'cudnn64_8.dll': 'cuDNN 8.x (Alternative)',
}

all_found = True
for dll_name, description in required_dlls.items():
    dll_path = os.path.join(cuda_bin, dll_name)
    exists = os.path.exists(dll_path)
    status = "✓" if exists else "✗"
    print(f"   {status} {dll_name:25s} - {description}")
    if not exists:
        all_found = False

print("\n3. Checking PATH Environment Variable...")
path_env = os.environ.get('PATH', '')
if cuda_bin in path_env:
    print(f"   ✓ CUDA bin is in PATH")
else:
    print(f"   ✗ CUDA bin is NOT in PATH")
    print(f"   Add this to PATH: {cuda_bin}")

print("\n4. Listing all CUDA DLLs...")
if os.path.exists(cuda_bin):
    dlls = [f for f in os.listdir(cuda_bin) if f.endswith('.dll')]
    print(f"   Found {len(dlls)} DLLs total")
    
    # Show important ones
    important = [d for d in dlls if any(x in d.lower() for x in ['cublas', 'cudnn', 'cudart'])]
    if important:
        print(f"   Important DLLs ({len(important)}):")
        for dll in sorted(important)[:15]:  # Show first 15
            print(f"     - {dll}")

print("\n5. Testing ONNX Runtime...")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"   Available providers: {providers}")
    
    if 'CUDAExecutionProvider' in providers:
        print("   ✓ CUDAExecutionProvider is available!")
    else:
        print("   ✗ CUDAExecutionProvider is NOT available")
        print("   This means CUDA DLLs are not accessible to Python")
except Exception as e:
    print(f"   ✗ Error importing onnxruntime: {e}")

print("\n" + "=" * 60)
if all_found:
    print("SUMMARY: All required DLLs found!")
    print("If CUDA still doesn't work, restart your terminal/IDE.")
else:
    print("SUMMARY: Some DLLs are missing!")
    print("\nSOLUTIONS:")
    if not os.path.exists(os.path.join(cuda_bin, 'cublas64_12.dll')):
        print("1. Reinstall CUDA Toolkit (Express Installation)")
        print("   https://developer.nvidia.com/cuda-downloads")
    if not os.path.exists(os.path.join(cuda_bin, 'cudnn64_9.dll')) and \
       not os.path.exists(os.path.join(cuda_bin, 'cudnn64_8.dll')):
        print("2. Install cuDNN:")
        print("   - Download from: https://developer.nvidia.com/cudnn")
        print("   - Extract and copy files to CUDA directory")
print("=" * 60)

