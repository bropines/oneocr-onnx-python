# How We Decrypted Microsoft OneOCR Models

## Or: Runtime API Hooking Without a Disassembler

**TL;DR**: Microsoft encrypts their OCR models inside `oneocr.onemodel`.
We extracted all 11 raw ONNX sub-models by intercepting the ONNX Runtime C API
*in memory* at the exact moment the DLL decrypts and loads them — entirely in
Python, no disassembler, no debugger, no kernel driver.

---

## Background

Windows 11 ships a proprietary OCR engine used by Snipping Tool, PowerToys
and other Microsoft apps. The engine lives in two files:

| File | Role |
|---|---|
| `oneocr.dll` | The OCR pipeline: text detection, script classification, recognition |
| `oneocr.onemodel` | An encrypted container holding 11 ONNX neural networks |
| `onnxruntime.dll` | Microsoft's own build of ONNX Runtime that the DLL links against |

The model file is encrypted with a key hardcoded into the DLL.  Without the
key, the `.onemodel` file is opaque binary noise.  Extracting the key and
reimplementing the decryption would require serious reverse engineering.

But there is a simpler observation: **the DLL must decrypt the models before
loading them**.  And when it loads them, it calls a well-known public C API:

```c
OrtStatus* CreateSessionFromArray(
    OrtEnv*        env,
    const void*    model_data,   // ← fully decrypted model bytes, right here
    size_t         model_data_length,
    OrtSessionOptions* options,
    OrtSession**   out
);
```

At the moment this function is called, `model_data` points to a plaintext
ONNX model in memory.  If we can intercept that call, we get the model for
free.

---

## The Attack Surface: `OrtGetApiBase`

ONNX Runtime exposes *all* its functionality through a single entry point:

```c
const OrtApiBase* OrtGetApiBase(void);
```

This returns a pointer to a struct:

```c
struct OrtApiBase {
    const OrtApi* (*GetApi)(uint32_t version);
    const char*   (*GetVersionString)(void);
};
```

`GetApi(version)` in turn returns a pointer to the main `OrtApi` struct — a
table of ~200 function pointers covering the entire ONNX Runtime C API.
`CreateSessionFromArray` is slot **#8** in that table.

The key insight: **if we replace what `OrtGetApiBase()` returns, we control
the entire API table**.

---

## The Hook — Step by Step

### 1. Load `onnxruntime.dll` before `oneocr.dll`

```python
lib_ort = ctypes.WinDLL("onnxruntime.dll")
```

By loading it ourselves first, our Python process holds the DLL handle.
When `oneocr.dll` later calls `LoadLibrary("onnxruntime.dll")`, Windows
returns the *already-loaded* instance — so both share the same code pages
in memory.

### 2. Read the real API table address

```python
orig_ort_get_api_base = lib_ort.OrtGetApiBase
orig_ort_get_api_base.restype = ctypes.POINTER(OrtApiBase)
real_api_base = orig_ort_get_api_base()          # call once to get real struct
```

We keep a reference to the original so we can still forward calls through it.

### 3. Patch `OrtGetApiBase` in memory to return our fake struct

```python
PAGE_EXECUTE_READWRITE = 0x40
VirtualProtect(fn_addr, 10, PAGE_EXECUTE_READWRITE, byref(old_protect))

# x86-64: mov rax, <64-bit address>; ret
patch = b'\x48\xB8' + struct.pack('<Q', fake_api_base_addr) + b'\xC3'
ctypes.memmove(fn_addr, patch, len(patch))
```

The 10-byte patch is:
```
48 B8 xx xx xx xx xx xx xx xx   MOV RAX, <addr of our fake OrtApiBase>
C3                               RET
```

Now any call to `OrtGetApiBase()` — from `oneocr.dll` or anyone else —
returns our fake struct instead of the real one.

### 4. Build the fake `OrtApiBase` and `OrtApi` table

Our `fake_get_api()` Python function is called whenever the DLL requests
the API table.  We:

1. Call the *original* `GetApi(version)` to get the real 4000-byte table.
2. Copy it into a new Python buffer.
3. **Overwrite slot #8** (`CreateSessionFromArray`) with our hook function pointer.
4. **Overwrite slot #151** (`CreateSessionFromArrayWithPrepackedWeightsContainer`)
   with a second hook for the alternative loading path.
5. Return the address of our modified buffer.

```python
def fake_get_api(version):
    orig_table_addr = real_api.GetApi(version)
    
    buf = ctypes.create_string_buffer(4000)
    ctypes.memmove(buf, orig_table_addr, 4000)  # copy real table

    # Replace slot 8
    hook_ptr = ctypes.cast(my_hook_csfa, ctypes.c_void_p).value
    buf[8*8 : 8*8+8] = struct.pack('<Q', hook_ptr)

    all_buffers.append(buf)   # ← critical: keep alive so GC doesn't free it
    return ctypes.cast(buf, ctypes.c_void_p).value
```

> **GC pitfall**: We store every allocated buffer in a list.  The DLL holds
> raw C pointers into Python-managed memory; if Python garbage-collects the
> buffer while the DLL is still running, you get an Access Violation.
> We discovered this the hard way on the first attempt.

### 5. The hook intercepts decrypted model bytes

```python
@ctypes.WINFUNCTYPE(ctypes.c_void_p,   # return: OrtStatus*
    ctypes.c_void_p,                   # OrtEnv*
    ctypes.c_void_p,                   # model_data   ← plaintext ONNX
    ctypes.c_size_t,                   # model_data_length
    ctypes.c_void_p,                   # OrtSessionOptions*
    ctypes.c_void_p)                   # OrtSession** out
def hook_create_session_from_array(env, model_data, model_data_length, options, out):
    # model_data now points to fully decrypted ONNX bytes in memory
    raw = ctypes.string_at(model_data, model_data_length)
    
    if model_data_length > 1_048_576:   # > 1 MB → real neural network
        with open(f"decrypted_model_{counter}_{model_data_length}.onnx", "wb") as f:
            f.write(raw)
    else:                               # small → vocab buffer
        with open(f"vocab_{counter}_{model_data_length}.bin", "wb") as f:
            f.write(raw)

    # Forward to the real function — oneocr.dll must still work normally
    return real_create_session_from_array(env, model_data, model_data_length, options, out)
```

### 6. Trigger the decryption

```python
lib_oneocr = ctypes.WinDLL("oneocr.dll")
lib_oneocr.CreateOcrInitOptions(byref(init_opts))
lib_oneocr.OcrInitOptionsSetUseModelDelayLoad(init_opts, 0)  # eager load all models
lib_oneocr.CreateOcrPipeline(model_path, key_buf, init_opts, byref(pipeline))
```

`CreateOcrPipeline` reads `oneocr.onemodel`, decrypts each sub-model, and
calls `CreateSessionFromArray` once per model.  Our hook fires 11 times
(plus ~22 more times for small vocab buffers) and saves everything to disk.

---

## What We Got

| # | File | Size | Role |
|---|---|---|---|
| 1 | `text_detector.onnx` | 11.0 MB | FPN text region detector |
| 2 | `script_classifier.onnx` | 3.3 MB | Script/language/orientation classifier |
| 3–11 | `recognizer_v*.onnx` | 1.7–12.8 MB | CRNN+CTC recognizers per script group |
| — | `vocab_*.bin` | ~27–29 KB each | Auxiliary ONNX sessions (vocab/lookup) |

All 11 files are valid, standard ONNX models loadable by any `onnxruntime`
installation on any OS — no `oneocr.dll`, no Windows dependency.

---

## Why This Works (and Is Hard to Prevent)

This is a **userland API interception** — not a kernel exploit, not a
vulnerability.  We are simply:

- Calling public Windows APIs (`VirtualProtect`, `LoadLibrary`).
- Patching memory *inside our own process* (fully legal, no privilege needed).
- Forwarding every real call so the DLL continues to function correctly.

Microsoft could make this harder by:
- **Verifying `OrtGetApiBase` integrity** before use (signature check, CRC on the code page).
- **Using a private ONNX Runtime build** with renamed entry points.
- **Loading models via a kernel driver** that never exposes plaintext to user space.

But all of those options carry significant engineering cost and break
legitimate Windows tooling.  For a component that lives inside a sandboxed
app container and whose *API* is intentionally public, encrypting the model
file is a reasonable deterrent against casual extraction — and that's exactly
what it is.

---

## Lessons Learned

1. **The decryption boundary is the attack surface.**  Any system that
   decrypts data in the same process that uses it can be tapped at the
   point of use.
2. **Public C APIs are stable hooking targets.**  The ONNX Runtime API
   table layout is versioned and documented.  Slot #8 won't move.
3. **Python + ctypes is surprisingly powerful** for low-level Windows
   interop: `VirtualProtect`, in-memory patching, function pointer
   replacement — all without writing a single line of C.
4. **Watch the garbage collector.**  When native code holds raw pointers
   into Python-allocated memory, you must keep Python references alive
   for the entire duration of the native call chain.

---

*Research performed for educational purposes.  
No DLL binaries or model files are distributed with this repository.*
