# /// script
# dependencies = [
#   "onnx>=1.14.0",
#   "onnxruntime>=1.16.0",
#   "numpy>=1.20.0",
#   "pillow>=9.0.0",
# ]
# ///
"""
OneOCR Extract-All Utility.

Fully automated script to:
1. Load onnxruntime.dll and hook OrtGetApiBase.
2. Load oneocr.dll and intercept decrypted ONNX models during initialization.
3. Dynamically intercept OrtApi::Run to inject greedy path CTC logits.
4. Auto-classify all decrypted models on disk based on graph parsing.
5. Reconstruct all vocabulary text files for CJK, Cyrillic, Latin, etc.
"""
import ctypes
import os
import sys
import struct
import io
import time
from ctypes import Structure, byref, POINTER, c_int64, c_int32, c_float, c_ubyte, c_char_p, c_void_p, c_size_t
from pathlib import Path

# Force UTF-8 stdout for clean terminal logs
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import onnx
except ImportError:
    print("Installing onnx dependency...")
    os.system(f"{sys.executable} -m pip install onnx -q")
    import onnx

BASE_DIR = Path(__file__).parent
BIN_DIR = BASE_DIR / "bin"
MODELS_DIR = BASE_DIR / "models"
VOCAB_DIR = MODELS_DIR / "vocab"
BUFFERS_DIR = BASE_DIR / "vocab_buffers"
RAW_DIR = MODELS_DIR / "raw_decrypted"

# Create folders
for d in [MODELS_DIR, VOCAB_DIR, BUFFERS_DIR, RAW_DIR, MODELS_DIR / "detector", MODELS_DIR / "classifier", MODELS_DIR / "recognizers"]:
    d.mkdir(parents=True, exist_ok=True)

if not (BIN_DIR / "oneocr.dll").exists():
    print(f"ERROR: oneocr.dll not found in {BIN_DIR}")
    print("Please run download_and_extract_oneocr.py first.")
    sys.exit(1)

# ---- Load DLLs -------------------------------------------------------------
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.SetDllDirectoryW(str(BIN_DIR))

print("Loading onnxruntime.dll...")
lib_ort = ctypes.WinDLL(str(BIN_DIR / "onnxruntime.dll"))

# ---- Hook variables --------------------------------------------------------
global_references = {}
hooked_callbacks = []
sessions = {}
recognizer_sizes_to_v = {}

target_script_id = 0
current_offset = 0
current_recognizer_v = None

# Address placeholders for original ORT API functions
orig_csfa_addr = None
orig_csfap_addr = None
orig_run_addr = None

get_tensor_mutable_data_fn = None
get_tensor_type_and_shape_fn = None
get_dimensions_count_fn = None
get_dimensions_fn = None
release_tensor_type_and_shape_info_fn = None

# Ort structures
class OrtApiBase(Structure):
    pass

GetApiFuncType = ctypes.WINFUNCTYPE(c_void_p, c_int32)
GetVersionStringFuncType = ctypes.WINFUNCTYPE(c_char_p)

OrtApiBase._fields_ = [
    ("GetApi", GetApiFuncType),
    ("GetVersionString", GetVersionStringFuncType)
]

# Hook prototypes
CreateSessionFromArrayType = ctypes.WINFUNCTYPE(c_void_p, c_void_p, c_void_p, c_size_t, c_void_p, c_void_p)
CreateSessionFromArrayPrepackedType = ctypes.WINFUNCTYPE(c_void_p, c_void_p, c_void_p, c_size_t, c_void_p, c_void_p, c_void_p)
RunType = ctypes.WINFUNCTYPE(c_void_p, c_void_p, c_void_p, POINTER(c_char_p), POINTER(c_void_p), c_size_t, POINTER(c_char_p), c_size_t, POINTER(c_void_p))

def save_raw_model(model_bytes: bytes, size: int) -> str:
    """Save raw model or vocab buffer quickly."""
    if size > 1024 * 1024:
        out_path = RAW_DIR / f"model_{size}.onnx"
        out_path.write_bytes(model_bytes)
        return f"model_{size}.onnx"
    else:
        out_path = RAW_DIR / f"vocab_{size}.bin"
        out_path.write_bytes(model_bytes)
        return f"vocab_{size}.bin"

# ---- Hooks implementation --------------------------------------------------
def hook_create_session_from_array(env, model_data, model_data_length, options, out):
    orig_func = CreateSessionFromArrayType(orig_csfa_addr)
    status = orig_func(env, model_data, model_data_length, options, out)
    if status is None or status == 0:
        try:
            session_ptr = ctypes.cast(out, POINTER(c_void_p)).contents.value
            if session_ptr:
                model_bytes = ctypes.string_at(model_data, model_data_length)
                name = save_raw_model(model_bytes, model_data_length)
                print(f"[DECRYPTED] session={session_ptr:#x} size={model_data_length:,} bytes -> {name}")
                sessions[session_ptr] = {
                    'size': model_data_length,
                    'V': None
                }
        except Exception as e:
            print("Error in hook_create_session_from_array:", e)
    return status

def hook_create_session_from_array_prepacked(env, model_data, model_data_length, options, prepacked, out):
    orig_func = CreateSessionFromArrayPrepackedType(orig_csfap_addr)
    status = orig_func(env, model_data, model_data_length, options, prepacked, out)
    if status is None or status == 0:
        try:
            session_ptr = ctypes.cast(out, POINTER(c_void_p)).contents.value
            if session_ptr:
                model_bytes = ctypes.string_at(model_data, model_data_length)
                name = save_raw_model(model_bytes, model_data_length)
                print(f"[DECRYPTED] session={session_ptr:#x} size={model_data_length:,} bytes (prepacked) -> {name}")
                sessions[session_ptr] = {
                    'size': model_data_length,
                    'V': None
                }
        except Exception as e:
            print("Error in hook_create_session_from_array_prepacked:", e)
    return status

def hook_run(session, run_options, input_names, inputs, input_len, output_names, output_len, outputs):
    orig_func = RunType(orig_run_addr)
    print(f"[hook_run] Calling original Run: session={session:#x}, input_len={input_len}, output_len={output_len}")
    print(f"[hook_run] input_names={input_names}, inputs={inputs}, output_names={output_names}, outputs={outputs}")
    try:
        status = orig_func(session, run_options, input_names, inputs, input_len, output_names, output_len, outputs)
    except Exception as e:
        print(f"[hook_run] orig_func crashed with: {e}")
        raise e
    print(f"[hook_run] orig_func returned status={status}")
    if status is not None and status != 0:
        return status
        
    try:
        if not outputs or not ctypes.cast(outputs, ctypes.c_void_p).value:
            return status
            
        info = sessions.get(session, {'size': 0, 'V': None})
        if not info['size']:
            return status
            
        # 1. Patch Classifier output to force target_script_id
        is_classifier = (info['size'] == 3456873 or output_len == 6)
        if is_classifier and output_len > 0:
            # Dynamically look up the index of 'script_id_score'
            script_id_out_idx = -1
            for idx in range(output_len):
                try:
                    name_bytes = output_names[idx]
                    if name_bytes:
                        name = name_bytes.decode('utf-8', errors='ignore')
                        if name == "script_id_score":
                            script_id_out_idx = idx
                            break
                except Exception:
                    pass
            
            if script_id_out_idx != -1 and outputs[script_id_out_idx]:
                try:
                    out_tensor = outputs[script_id_out_idx]
                    data_ptr = c_void_p()
                    get_tensor_mutable_data_fn(out_tensor, byref(data_ptr))
                    if data_ptr.value:
                        import numpy as np
                        arr = np.ctypeslib.as_array(ctypes.cast(data_ptr, POINTER(c_float)), shape=(1, 1, 10))
                        arr.fill(-999.0)
                        arr[0, 0, target_script_id] = 10.0
                except Exception as e:
                    print("Error patching classifier output:", e)

                
        # 2. Patch Recognizer output to inject sequence of vocab indices
        is_recognizer = (output_len == 1 and info['size'] > 1024 * 1024)
        if is_recognizer and outputs[0]:
            global current_recognizer_v
            try:
                out_tensor = outputs[0]
                
                info_ptr = c_void_p()
                get_tensor_type_and_shape_fn(out_tensor, byref(info_ptr))
                if info_ptr.value:
                    num_dims = c_size_t()
                    get_dimensions_count_fn(info_ptr, byref(num_dims))
                    dims = (c_int64 * num_dims.value)()
                    get_dimensions_fn(info_ptr, dims, num_dims.value)
                    release_tensor_type_and_shape_info_fn(info_ptr)
                    
                    if num_dims.value >= 3:
                        T, _, V = list(dims)[:3]
                        current_recognizer_v = V
                        info['V'] = V
                        recognizer_sizes_to_v[info['size']] = V
                        
                        data_ptr = c_void_p()
                        get_tensor_mutable_data_fn(out_tensor, byref(data_ptr))
                        if data_ptr.value:
                            import numpy as np
                            arr = np.ctypeslib.as_array(ctypes.cast(data_ptr, POINTER(c_float)), shape=(T, 1, V))
                            arr.fill(-999.0)
                            limit = min(T - 2, 200)
                            for t in range(1, limit + 1):
                                idx = current_offset + t
                                if idx < V:
                                    arr[t, 0, idx] = 0.0
            except Exception as e:
                print("Error patching recognizer output:", e)
    except Exception as e:
        print("General error in hook_run:", e)


        
    return status


# ---- Hook injection setup --------------------------------------------------
orig_ort_get_api_base = lib_ort.OrtGetApiBase
orig_ort_get_api_base.restype = POINTER(OrtApiBase)
orig_ort_get_api_base.argtypes = []

orig_api_base = orig_ort_get_api_base().contents
real_api_ptr = orig_api_base.GetApi(7)

# Retrieve original function pointers from slots
orig_csfa_addr = ctypes.cast(real_api_ptr + 8*8, POINTER(c_void_p)).contents.value
orig_run_addr = ctypes.cast(real_api_ptr + 9*8, POINTER(c_void_p)).contents.value
orig_csfap_addr = ctypes.cast(real_api_ptr + 151*8, POINTER(c_void_p)).contents.value

# Helper to bind ctypes function from api pointer
def get_api_fn(api_ptr, idx, argtypes, restype):
    fn_ptr = ctypes.cast(api_ptr + idx * 8, POINTER(c_void_p)).contents.value
    return ctypes.WINFUNCTYPE(restype, *argtypes)(fn_ptr)

get_tensor_mutable_data_fn = get_api_fn(real_api_ptr, 51, [c_void_p, POINTER(c_void_p)], c_void_p)
get_tensor_type_and_shape_fn = get_api_fn(real_api_ptr, 65, [c_void_p, POINTER(c_void_p)], c_void_p)
get_dimensions_count_fn = get_api_fn(real_api_ptr, 61, [c_void_p, POINTER(c_size_t)], c_void_p)
get_dimensions_fn = get_api_fn(real_api_ptr, 62, [c_void_p, POINTER(c_int64), c_size_t], c_void_p)
release_tensor_type_and_shape_info_fn = get_api_fn(real_api_ptr, 99, [c_void_p], None)

# Configure VirtualProtect
kernel32.VirtualProtect.argtypes = [c_void_p, c_size_t, ctypes.c_ulong, POINTER(ctypes.c_ulong)]
kernel32.VirtualProtect.restype = ctypes.c_int

old_protect = ctypes.c_ulong()
PAGE_EXECUTE_READWRITE = 0x40

# In-place patch the API table
if not kernel32.VirtualProtect(real_api_ptr, 1600, PAGE_EXECUTE_READWRITE, ctypes.byref(old_protect)):
    raise RuntimeError("VirtualProtect on API table failed")

cb_csfa = CreateSessionFromArrayType(hook_create_session_from_array)
hooked_callbacks.append(cb_csfa)
ctypes.cast(real_api_ptr + 8*8, POINTER(c_void_p))[0] = ctypes.cast(cb_csfa, ctypes.c_void_p).value

cb_run = RunType(hook_run)
hooked_callbacks.append(cb_run)
ctypes.cast(real_api_ptr + 9*8, POINTER(c_void_p))[0] = ctypes.cast(cb_run, ctypes.c_void_p).value

cb_csfap = CreateSessionFromArrayPrepackedType(hook_create_session_from_array_prepacked)
hooked_callbacks.append(cb_csfap)
ctypes.cast(real_api_ptr + 151*8, POINTER(c_void_p))[0] = ctypes.cast(cb_csfap, ctypes.c_void_p).value

kernel32.VirtualProtect(real_api_ptr, 1600, old_protect.value, ctypes.byref(old_protect))

print("ONNX Runtime API hooks patched in-place successfully.")

# ---- Post processing classification ---------------------------------------
def post_process_models():
    print("\n" + "=" * 70)
    print("Post-processing and organizing decrypted models...")
    print("=" * 70)
    
    if not RAW_DIR.exists():
        print("No raw decrypted models found.")
        return
        
    for item in RAW_DIR.iterdir():
        if not item.is_file():
            continue
            
        if item.name.endswith(".bin"):
            dest = BUFFERS_DIR / item.name
            if dest.exists():
                dest.unlink()
            item.rename(dest)
            print(f"Organized: {item.name} -> vocab_buffers/")
            continue
            
        if item.name.endswith(".onnx"):
            try:
                model_bytes = item.read_bytes()
                model = onnx.load_model_from_string(model_bytes)
                graph = model.graph
                output_names = [out.name for out in graph.output]
                
                # 1. Text Detector
                if any("fpn" in name for name in output_names):
                    dest = MODELS_DIR / "detector" / "text_detector.onnx"
                    if dest.exists():
                        dest.unlink()
                    item.rename(dest)
                    print(f"Organized: {item.name} -> models/detector/text_detector.onnx")
                    
                # 2. Script/Language Classifier
                elif "script_id_score" in output_names:
                    dest = MODELS_DIR / "classifier" / "script_classifier.onnx"
                    if dest.exists():
                        dest.unlink()
                    item.rename(dest)
                    print(f"Organized: {item.name} -> models/classifier/script_classifier.onnx")
                    
                # 3. Text Recognizer
                elif "logsoftmax" in output_names:
                    logsoftmax_out = next((out for out in graph.output if out.name == "logsoftmax"), None)
                    V = 0
                    if logsoftmax_out:
                        try:
                            V = logsoftmax_out.type.tensor_type.shape.dim[-1].dim_value
                        except Exception:
                            pass
                    if V == 0:
                        V = recognizer_sizes_to_v.get(item.stat().st_size, 415)
                        
                    # Map vocab sizes to descriptive language names
                    name_map = {
                        32632: "cjk",
                        548: "cyrillic",
                        415: "latin",
                        221: "arabic",
                        237: "devanagari",
                        244: "hebrew",
                        199: "thai",
                        201: "bengali",
                        179: "greek"
                    }
                    lang_name = name_map.get(V, f"v{V}")
                    dest = MODELS_DIR / "recognizers" / f"recognizer_{lang_name}.onnx"
                    if dest.exists():
                        dest.unlink()
                    item.rename(dest)
                    print(f"Organized: {item.name} -> models/recognizers/recognizer_{lang_name}.onnx")
                else:
                    dest = BUFFERS_DIR / item.name
                    if dest.exists():
                        dest.unlink()
                    item.rename(dest)
                    print(f"Organized: {item.name} -> vocab_buffers/")
            except Exception as e:
                print(f"Failed to post-process {item.name}: {e}")
                
    try:
        RAW_DIR.rmdir()
    except Exception:
        pass

# ---- Main Execution --------------------------------------------------------
def main():
    print("Loading oneocr.dll...")
    dll = ctypes.WinDLL(str(BIN_DIR / "oneocr.dll"))
    
    dll.CreateOcrInitOptions.restype = c_int64
    dll.CreateOcrInitOptions.argtypes = [POINTER(c_int64)]
    dll.OcrInitOptionsSetUseModelDelayLoad.restype = c_int64
    dll.OcrInitOptionsSetUseModelDelayLoad.argtypes = [c_int64, ctypes.c_char]
    dll.CreateOcrPipeline.restype = c_int64
    dll.CreateOcrPipeline.argtypes = [c_char_p, c_char_p, c_int64, POINTER(c_int64)]
    dll.ReleaseOcrPipeline.restype = None
    dll.ReleaseOcrPipeline.argtypes = [c_int64]
    dll.ReleaseOcrInitOptions.restype = None
    dll.ReleaseOcrInitOptions.argtypes = [c_int64]

    dll.CreateOcrProcessOptions.restype = c_int64
    dll.CreateOcrProcessOptions.argtypes = [POINTER(c_int64)]
    dll.ReleaseOcrProcessOptions.restype = None
    dll.ReleaseOcrProcessOptions.argtypes = [c_int64]
    dll.ReleaseOcrResult.restype = None
    dll.ReleaseOcrResult.argtypes = [c_int64]

    dll.GetOcrLineCount.restype = c_int64
    dll.GetOcrLineCount.argtypes = [c_int64, POINTER(c_int64)]
    dll.GetOcrLine.restype = c_int64
    dll.GetOcrLine.argtypes = [c_int64, c_int64, POINTER(c_int64)]
    dll.GetOcrLineContent.restype = c_int64
    dll.GetOcrLineContent.argtypes = [c_int64, POINTER(c_char_p)]

    class ImageStructure(Structure):
        _fields_ = [
            ('type', c_int32),
            ('width', c_int32),
            ('height', c_int32),
            ('_reserved', c_int32),
            ('step_size', c_int64),
            ('data_ptr', POINTER(c_ubyte))
        ]


    dll.RunOcrPipeline.restype = c_int64
    dll.RunOcrPipeline.argtypes = [c_int64, POINTER(ImageStructure), c_int64, POINTER(c_int64)]

    print("\nInitializing OCR pipeline (triggers models decryption)...")
    init_opts = c_int64()
    dll.CreateOcrInitOptions(byref(init_opts))
    dll.OcrInitOptionsSetUseModelDelayLoad(init_opts, 0) # eager load
    
    model_path = str(BIN_DIR / "oneocr.onemodel").encode()
    key = b'kj)TGtrK>f]b[Piow.gU+nC@s""""""4'
    pipeline = c_int64()
    ret = dll.CreateOcrPipeline(model_path, key, init_opts, byref(pipeline))
    if ret != 0:
        print(f"Failed to create pipeline: {ret:#x}")
        sys.exit(1)
        
    print("\nAll ONNX models successfully decrypted and saved.")
    
    # ── Reconstruct character vocabularies using feedback loop ──
    print("\nStarting vocabulary extraction loop...")
    
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGBA', (2540, 200), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except Exception:
        font = None
    draw.text((30, 60), "Это очень длинный текст для перехвата сессии распознавания", fill=(0, 0, 0, 255), font=font)
    
    b, g, r, a = img.split()
    bgra = Image.merge('RGBA', (b, g, r, a))
    raw_bytes = bgra.tobytes()
    arr = (c_ubyte * len(raw_bytes)).from_buffer_copy(raw_bytes)
    img_struct = ImageStructure(type=3, width=bgra.width, height=bgra.height, _reserved=0, step_size=bgra.width*4, data_ptr=arr)

    proc_opts = c_int64()
    dll.CreateOcrProcessOptions(byref(proc_opts))
    
    global target_script_id, current_offset
    extracted_vocabs = 0
    
    # Map vocab sizes to descriptive language names
    name_map = {
        32632: "cjk",
        548: "cyrillic",
        415: "latin",
        221: "arabic",
        237: "devanagari",
        244: "hebrew",
        199: "thai",
        201: "bengali",
        179: "greek"
    }

    # Iterate through all 10 script IDs
    for script_id in range(10):
        target_script_id = script_id
        current_offset = 0
        captured_vocab = []
        V_limit = None
        
        print(f"\nScanning Script ID {script_id}...")
        
        while True:
            global current_recognizer_v
            current_recognizer_v = None
            
            result_h = c_int64()
            ret = dll.RunOcrPipeline(pipeline, byref(img_struct), proc_opts, byref(result_h))
            
            if ret == 0 and result_h.value != 0:
                line_count = c_int64()
                dll.GetOcrLineCount(result_h, byref(line_count))
                
                decoded_text = ""
                if line_count.value > 0:
                    line_h = c_int64()
                    dll.GetOcrLine(result_h, 0, byref(line_h))
                    text_ptr = c_char_p()
                    dll.GetOcrLineContent(line_h, byref(text_ptr))
                    if text_ptr.value:
                        decoded_text = text_ptr.value.decode('utf-8', errors='replace')
                
                dll.ReleaseOcrResult(result_h)
                
                if current_recognizer_v is None:
                    break
                    
                V_limit = current_recognizer_v
                chunk_size = min(630 - 2, 200)
                
                while len(captured_vocab) < V_limit:
                    captured_vocab.append("")
                    
                for k, char in enumerate(decoded_text):
                    idx = current_offset + 1 + k
                    if idx < V_limit:
                        captured_vocab[idx] = char
                        
                current_offset += chunk_size
                if current_offset >= V_limit - 1:
                    extracted_vocabs += 1
                    lang_name = name_map.get(V_limit, f"v{V_limit}")
                    out_path = VOCAB_DIR / f"vocab_{lang_name}.txt"
                    with open(out_path, "w", encoding="utf-8") as f:
                        for idx, char in enumerate(captured_vocab):
                            f.write(f"{idx}: {char}\n")
                    print(f"  -> Extracted vocab for {lang_name} ({len(captured_vocab)} entries) -> models/vocab/vocab_{lang_name}.txt")
                    break
            else:
                break
                
    # Cleanup
    dll.ReleaseOcrProcessOptions(proc_opts)
    dll.ReleaseOcrPipeline(pipeline)
    dll.ReleaseOcrInitOptions(init_opts)
    
    # Organize raw files
    post_process_models()
    
    print("\n" + "=" * 70)
    print(f"SUCCESS: Decrypted models and extracted {extracted_vocabs} vocabularies.")
    print(f"All outputs are located in the models/ folder.")
    print("=" * 70)

if __name__ == "__main__":
    main()
