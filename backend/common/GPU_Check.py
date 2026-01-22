import torch
import onnxruntime as ort  # ONNX Runtime for execution provider detection

#  Function to print MPS (Metal Performance Shaders) info
def print_MPS():
    """Prints PyTorch version and MPS availability (for macOS)."""
    print("PyTorch Version:", torch.__version__)
    print("MPS (Metal) Available:", torch.backends.mps.is_available())
    print("MPS Built:", torch.backends.mps.is_built())
    print("CUDA Available:", torch.cuda.is_available())

#  Function to check if a GPU (CUDA or MPS) is available
def is_GPU_available():
    """
    Returns True if any GPU (CUDA or MPS) is available, else False.
    """
    return torch.cuda.is_available() or torch.backends.mps.is_available()

#  Function to get the best available PyTorch device (CUDA, MPS, or CPU)
def get_device():
    """
    Returns the best available device for PyTorch.
    """
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

#  Function to get the best available ONNX Execution Provider
def get_onnx_provider():
    onnx_providers = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in onnx_providers:
        providers.append("CUDAExecutionProvider")
    if "CoreMLExecutionProvider" in onnx_providers:
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")  # Always keep CPU as a fallback
    
    print(f" ONNX Execution Providers Selected: {providers}")
    return providers


#  Function to move a model to MPS (for macOS users)
def to_mps(model):
    """
    Moves the model to MPS (Metal Performance Shaders) on Mac M1/M2.
    Defaults to CPU if MPS is unavailable.
    """
    if torch.backends.mps.is_available():
        print("To MPS")
        return model.to("mps")
    else:
        print("MPS is not available. Running on CPU instead.")
        return model.to("cpu")
