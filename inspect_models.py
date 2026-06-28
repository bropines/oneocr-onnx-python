"""
Inspect all decrypted ONNX models: metadata, graph structure,
inputs/outputs, operator types, and weight tensor names.

Usage:
    python inspect_models.py [model_path]   # single model
    python inspect_models.py               # all models in models/
"""
import sys
import io
import os
from pathlib import Path
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("Installing onnx + onnxruntime...")
    os.system(f"{sys.executable} -m pip install onnx onnxruntime -q")
    import onnx
    import onnxruntime as ort

MODELS_DIR = Path(__file__).parent / "models"

def fmt_shape(shape) -> str:
    return "[" + ", ".join(str(d.dim_value) if d.dim_value else d.dim_param or "?"
                           for d in shape.dim) + "]"

def dtype_name(dtype_int: int) -> str:
    return onnx.TensorProto.DataType.Name(dtype_int)

def inspect_model(path: Path):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"MODEL: {path.relative_to(MODELS_DIR.parent)}")
    print(f"Size : {path.stat().st_size / 1024 / 1024:.2f} MB")
    print(sep)

    # ---------- onnx graph-level info ----------
    model = onnx.load(str(path))
    graph = model.graph

    # Metadata
    print(f"\n[Metadata]")
    print(f"  IR version       : {model.ir_version}")
    print(f"  Producer         : {model.producer_name or '—'}  v{model.producer_version or '—'}")
    print(f"  Domain           : {model.domain or '—'}")
    print(f"  Model version    : {model.model_version}")
    print(f"  Doc string       : {model.doc_string[:120] or '—'}")
    if model.metadata_props:
        for kv in model.metadata_props:
            print(f"  Meta [{kv.key}] = {kv.value[:80]}")

    # Graph summary
    print(f"\n[Graph]  name={graph.name or '—'}")
    print(f"  Nodes      : {len(graph.node)}")
    print(f"  Initializers: {len(graph.initializer)}  (named weight tensors)")

    # Inputs
    print(f"\n[Inputs]")
    for inp in graph.input:
        t = inp.type.tensor_type
        print(f"  '{inp.name}'  dtype={dtype_name(t.elem_type)}  shape={fmt_shape(t.shape)}")

    # Outputs
    print(f"\n[Outputs]")
    for out in graph.output:
        t = out.type.tensor_type
        print(f"  '{out.name}'  dtype={dtype_name(t.elem_type)}  shape={fmt_shape(t.shape)}")

    # Operator histogram
    op_counts = Counter(n.op_type for n in graph.node)
    print(f"\n[Operators]  ({len(op_counts)} unique)")
    for op, count in op_counts.most_common():
        print(f"  {count:4d}×  {op}")

    # Weight tensor names (initializers)
    if graph.initializer:
        print(f"\n[Initializer names]  (first 20 of {len(graph.initializer)})")
        for init in list(graph.initializer)[:20]:
            shape = list(init.dims)
            numel = 1
            for d in shape: numel *= d
            print(f"  '{init.name}'  shape={shape}  dtype={dtype_name(init.data_type)}  params={numel:,}")

    # ---------- onnxruntime session info ----------
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(path), sess_options=opts,
                                providers=['CPUExecutionProvider'])

    print(f"\n[OnnxRuntime session inputs]")
    for i in sess.get_inputs():
        print(f"  '{i.name}'  shape={i.shape}  type={i.type}")

    print(f"[OnnxRuntime session outputs]")
    for o in sess.get_outputs():
        print(f"  '{o.name}'  shape={o.shape}  type={o.type}")


def main():
    if len(sys.argv) > 1:
        paths = [Path(sys.argv[1])]
    else:
        paths = sorted(MODELS_DIR.rglob("*.onnx"))
        if not paths:
            print(f"No .onnx models found under {MODELS_DIR}")
            print("Run decrypt_oneocr_models.py first.")
            sys.exit(1)

    print(f"Inspecting {len(paths)} model(s)...")
    for p in paths:
        inspect_model(p)

    print(f"\n{'=' * 70}\nDone.")

if __name__ == "__main__":
    main()
