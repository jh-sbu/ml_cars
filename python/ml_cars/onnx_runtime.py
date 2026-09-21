"""Print the shared-library path from an installed CPU onnxruntime wheel.

Setup only: native execution subsequently loads this library without Python.
Usage: export ORT_DYLIB_PATH="$(python -m ml_cars.onnx_runtime)"
Alternatively, point ORT_DYLIB_PATH to a separately installed native runtime.
"""
import importlib.util
from pathlib import Path


def library_path():
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or spec.origin is None:
        raise RuntimeError("install ml-cars[onnx] or provide a native ONNX Runtime library")
    directory = Path(spec.origin).parent / "capi"
    candidates = sorted({p.resolve() for pattern in ("libonnxruntime.so*", "libonnxruntime*.dylib", "onnxruntime.dll") for p in directory.glob(pattern)})
    if len(candidates) != 1:
        raise RuntimeError(f"expected one ONNX Runtime shared library in {directory}, found {len(candidates)}")
    return candidates[0]


if __name__ == "__main__":
    print(library_path())
