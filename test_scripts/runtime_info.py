"""Execution-provider selection and startup diagnostics for ONNX Runtime on Jetson.

Split out of ``object_detection.py`` so the provider/versioning logic can be
tested and reused without pulling in OpenCV, the ZED SDK, or the detector.

Responsibilities
----------------
* Build a correctly-configured provider chain (TensorRT -> CUDA -> CPU).
* Create the ``InferenceSession`` and confirm which provider actually claimed it.
* Print a startup banner: available providers, selected provider, GPU name,
  CUDA / cuDNN / TensorRT versions, and whether real GPU kernels ran.
* Refuse to silently run on the CPU when the GPU was requested (``REQUIRE_GPU``),
  printing the concrete reason and the command that fixes it.
"""

import glob
import json
import os
import re
import subprocess
import tempfile

import onnxruntime as ort

# ------------------------
# Configuration
# ------------------------

# Hard-fail instead of silently running on the CPU. A CPU fallback on the Orin
# Nano costs ~7x inference latency, which is a bug, not a degraded mode.
REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "1") == "1"

# TensorRT is the fastest steady-state option (~10 ms/frame vs ~18 ms on CUDA)
# but is OFF by default: building the engine on this 8 GB Orin Nano takes ~8
# minutes, pushes the board into swap, and can destabilise it when the ZED SDK
# and other tools are already resident. Opt in deliberately with USE_TENSORRT=1,
# ideally on an otherwise idle board; once the engine is cached in TRT_CACHE_DIR
# later starts reuse it.
USE_TENSORRT = os.environ.get("USE_TENSORRT", "0") == "1"

# Prefer a prebuilt .engine file loaded directly through the TensorRT API. This
# is the fastest path and costs no build time, but only works if the engine was
# built for this exact GPU architecture and TensorRT version. Set
# USE_NATIVE_TRT=0 to ignore any engine file and go through ONNX Runtime.
USE_NATIVE_TRT = os.environ.get("USE_NATIVE_TRT", "1") == "1"
TRT_FP16 = os.environ.get("TRT_FP16", "1") == "1"

# Built engines are cached here, keyed by model + shape + TRT version, so only
# the first run pays the build cost.
TRT_CACHE_DIR = os.environ.get(
    "TRT_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".trt_cache"),
)

# Orin Nano has 8 GB of *unified* memory shared with ZED, RViz and QGC. Cap both
# the TRT build workspace and the ORT CUDA arena so inference cannot starve them.
TRT_WORKSPACE_BYTES = int(os.environ.get("TRT_WORKSPACE_BYTES", 512 * 1024 * 1024))
GPU_MEM_LIMIT_BYTES = int(os.environ.get("GPU_MEM_LIMIT_BYTES", 1024 * 1024 * 1024))

GPU_PROVIDERS = ("TensorrtExecutionProvider", "CUDAExecutionProvider")


# ------------------------
# Version probing
# ------------------------

def _run(cmd):
    """Best-effort command capture; returns '' instead of raising."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""


def gpu_name():
    """Human-readable GPU / board name.

    ``nvidia-smi`` does not report a useful name for Tegra iGPUs, so fall back to
    the device-tree board model, which is what identifies an Orin Nano.
    """
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).strip()
    if out and "not supported" not in out.lower():
        return out
    try:
        with open("/proc/device-tree/model", "rb") as f:
            return f.read().decode(errors="ignore").strip("\x00").strip()
    except OSError:
        return "unknown"


def cuda_version():
    """CUDA toolkit version string, or 'unknown'."""
    m = re.search(r"release (\d+\.\d+)", _run(["nvcc", "--version"]))
    if m:
        return m.group(1)
    try:
        with open("/usr/local/cuda/version.json") as f:
            return json.load(f)["cuda"]["version"]
    except Exception:
        return "unknown"


def cudnn_version():
    """cuDNN version taken from the installed shared-object soname."""
    libs = sorted(glob.glob("/usr/lib/*/libcudnn.so.*.*.*"))
    return libs[-1].split("libcudnn.so.")[-1] if libs else "unknown"


def tensorrt_version():
    """TensorRT version taken from the installed libnvinfer soname."""
    libs = sorted(glob.glob("/usr/lib/*/libnvinfer.so.*.*.*"))
    return libs[-1].split("libnvinfer.so.")[-1] if libs else "unknown"


def jetpack_version():
    """L4T release line from /etc/nv_tegra_release, or 'unknown'."""
    try:
        with open("/etc/nv_tegra_release") as f:
            return f.readline().strip().lstrip("# ")
    except OSError:
        return "unknown"


# ------------------------
# Provider chain
# ------------------------

def build_provider_chain():
    """Provider list for ``InferenceSession``, best-first, filtered to what exists.

    Each entry is ``(name, options)``. Options matter a lot on Jetson:

    * ``trt_engine_cache_enable`` — without it every process start rebuilds the
      engine (minutes).
    * ``trt_fp16_enable`` — roughly 2x throughput on Orin's tensor cores. YOLOv8
      detection heads tolerate fp16; box coordinates stay well inside its range.
    * ``cudnn_conv_algo_search=HEURISTIC`` — the default EXHAUSTIVE benchmarks
      every convolution algorithm at session init, burning seconds and transient
      VRAM for no steady-state gain.
    * ``arena_extend_strategy=kSameAsRequested`` — the default doubles the arena
      on each extension, which fragments and over-reserves unified memory.
    """
    available = set(ort.get_available_providers())
    chain = []

    if USE_TENSORRT and "TensorrtExecutionProvider" in available:
        os.makedirs(TRT_CACHE_DIR, exist_ok=True)
        chain.append(("TensorrtExecutionProvider", {
            "trt_fp16_enable": TRT_FP16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": TRT_CACHE_DIR,
            "trt_timing_cache_enable": True,
            "trt_max_workspace_size": TRT_WORKSPACE_BYTES,
            # TRT falls back to CUDA/CPU for unsupported subgraphs on its own.
            "trt_max_partition_iterations": 10,
        }))

    if "CUDAExecutionProvider" in available:
        chain.append(("CUDAExecutionProvider", {
            "device_id": 0,
            "cudnn_conv_algo_search": "HEURISTIC",
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": GPU_MEM_LIMIT_BYTES,
            "do_copy_in_default_stream": True,
        }))

    chain.append(("CPUExecutionProvider", {}))
    return chain


def session_options(enable_profiling=False):
    """ORT session options tuned for a 6-core Orin Nano.

    Inference is one small graph per frame; ORT's default thread pool spawns one
    thread per core and spends more time synchronising than computing, and it
    competes with the ZED depth pipeline for cores. On GPU the CPU threads only
    run the few fallback nodes, so 2 is plenty.
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    # Free initializer memory once the graph is built instead of holding both
    # the ORT copy and the protobuf copy resident.
    so.enable_mem_pattern = True
    so.log_severity_level = 3  # warnings only; suppresses per-node spam
    if enable_profiling:
        # Only for the warmup run; stopped immediately after via end_profiling().
        so.enable_profiling = True
        so.profile_file_prefix = os.path.join(tempfile.gettempdir(), "ort_startup")
    return so


class GpuUnavailable(RuntimeError):
    """Raised when GPU execution was required but could not be obtained."""


_FIX_HINT = (
    "Install the Jetson GPU build of ONNX Runtime (the PyPI 'onnxruntime' wheel\n"
    "  is CPU-only and has no CUDA/TensorRT provider compiled in):\n\n"
    "    pip uninstall -y onnxruntime onnxruntime-gpu\n"
    "    pip install onnxruntime-gpu --extra-index-url \\\n"
    "        https://pypi.jetson-ai-lab.io/jp6/cu126\n\n"
    "  Then re-run. Set REQUIRE_GPU=0 to allow CPU execution anyway."
)


def _gpu_node_fraction(session):
    """Fraction of executed graph nodes that ran on a GPU provider.

    Proof that CUDA/TensorRT kernels really executed, rather than the session
    merely *reporting* a GPU provider while every node quietly fell back to the
    CPU. Reads the profile of the warmup run already performed on ``session``,
    then deletes the trace file.

    Must be called on the live session — profiling a *second* session built from
    ``get_providers()`` would lose the provider options (notably the TensorRT
    engine-cache path) and trigger a full multi-minute engine rebuild.
    """
    try:
        path = session.end_profiling()   # stops profiling, flushes the trace
        if not path or not os.path.exists(path):
            return None
        with open(path) as f:
            events = json.load(f)
        os.remove(path)
        nodes = [e for e in events
                 if e.get("cat") == "Node" and e.get("args", {}).get("provider")]
        if not nodes:
            return None
        gpu = sum(1 for e in nodes
                  if e["args"]["provider"] in GPU_PROVIDERS)
        return gpu / len(nodes)
    except Exception:
        return None


def _zero_feed(session):
    """Build a zero-filled input dict matching the model's declared input shapes.

    Used for warmup (and kernel verification) so callers do not need a real frame
    before the session is usable.
    """
    import numpy as np
    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        dtype = np.float16 if "float16" in inp.type else np.float32
        feed[inp.name] = np.zeros(shape, dtype)
    return feed


def _try_native_trt():
    """Load a prebuilt ``.engine`` via the native TensorRT backend, or return None.

    Preferred over the ONNX Runtime TensorRT provider when an engine file already
    exists: ORT can only load engines it built itself, so pointing it at this
    model would kick off an ~8 minute on-device build. Loading the engine
    directly takes about a second and skips the ORT graph, session and CUDA arena
    entirely. Any failure here is non-fatal — the caller falls back to ORT.
    """
    if not USE_NATIVE_TRT:
        return None, None
    try:
        import trt_backend
    except ImportError as exc:            # tensorrt / cuda-python not installed
        return None, f"native TensorRT unavailable ({exc})"

    path = trt_backend.default_engine_path()
    if not os.path.exists(path):
        return None, f"no prebuilt engine at {path}"
    try:
        return trt_backend.TensorRTSession(path), None
    except Exception as exc:              # wrong arch / TRT version / corrupt file
        return None, f"prebuilt engine rejected: {exc}"


def create_session(model_path, warmup=True, verify_kernels=True):
    """Create an inference session on the fastest available backend and report it.

    Order of preference:

    1. Native TensorRT against a prebuilt ``.engine`` (fastest, smallest, no build).
    2. ONNX Runtime with the TensorRT provider (only if ``USE_TENSORRT=1``).
    3. ONNX Runtime with the CUDA provider.
    4. CPU — rejected unless ``REQUIRE_GPU=0``.

    Args:
        model_path: path to the .onnx file.
        warmup: run one zero-input inference so TensorRT engine loading and
            cuDNN plan selection happen here rather than on the first live frame.
        verify_kernels: run one profiled inference and report what fraction of
            graph nodes actually executed on a GPU provider.

    Returns:
        (session, info) where ``info`` is a dict of everything printed.

    Raises:
        GpuUnavailable: if ``REQUIRE_GPU`` and the session landed on the CPU.
    """
    available = ort.get_available_providers()

    # 1. Prebuilt TensorRT engine, if there is one and it loads.
    native, native_note = _try_native_trt()
    if native is not None:
        desc = native.describe()
        info = {
            "ort_version": "",   # ORT is not in the path at all here
            "ort_package": f"native TensorRT engine ({os.path.basename(desc['engine_path'])})",
            "available_providers": available,
            "requested_providers": ["NativeTensorRT"],
            "selected_provider": "NativeTensorRT",
            "on_gpu": True,
            "gpu_name": gpu_name(),
            "jetpack": jetpack_version(),
            "cuda": cuda_version(),
            "cudnn": cudnn_version(),
            "tensorrt": desc["tensorrt"],
            # A serialized engine is fp16 only if it was built that way; the flag
            # is the builder's choice, not ours, so report it as unknown here.
            "fp16": None,
            # Every layer of a TensorRT engine runs on the GPU by construction.
            "gpu_node_fraction": 1.0,
            "engine_device_mem_mb": desc["device_mem_mb"],
        }
        print_banner(info)
        return native, info

    chain = build_provider_chain()
    requested = [name for name, _ in chain]
    if native_note:
        print(f"[runtime] {native_note}; falling back to ONNX Runtime")

    # Profiling is enabled only when we intend to inspect node placement; it is
    # turned off again by end_profiling() inside _gpu_node_fraction().
    session = ort.InferenceSession(model_path, session_options(verify_kernels),
                                   providers=chain)
    active = session.get_providers()
    selected = active[0] if active else "none"
    on_gpu = selected in GPU_PROVIDERS

    gpu_frac = None
    if warmup or verify_kernels:
        # One zero-input run: builds/loads the TRT engine and picks cuDNN plans
        # here instead of on the first live frame, and produces the trace that
        # verify_kernels reads.
        session.run(None, _zero_feed(session))
        if verify_kernels:
            gpu_frac = _gpu_node_fraction(session)

    info = {
        "ort_version": ort.__version__,
        "ort_package": "onnxruntime-gpu" if set(available) & set(GPU_PROVIDERS)
                       else "onnxruntime (CPU-only build)",
        "available_providers": available,
        "requested_providers": requested,
        "selected_provider": selected,
        "on_gpu": on_gpu,
        "gpu_name": gpu_name(),
        "jetpack": jetpack_version(),
        "cuda": cuda_version(),
        "cudnn": cudnn_version(),
        "tensorrt": tensorrt_version() if selected == "TensorrtExecutionProvider" else None,
        "fp16": TRT_FP16 if selected == "TensorrtExecutionProvider" else False,
        "gpu_node_fraction": gpu_frac,
    }

    print_banner(info)

    if REQUIRE_GPU and not on_gpu:
        reason = ("no CUDA/TensorRT execution provider is compiled into this "
                  "onnxruntime build" if not set(available) & set(GPU_PROVIDERS)
                  else "the GPU provider failed to initialise and ORT fell back")
        raise GpuUnavailable(
            f"GPU inference required but unavailable: {reason}.\n\n  {_FIX_HINT}"
        )

    return session, info


def print_banner(info):
    """Print the startup diagnostics block."""
    bar = "=" * 62
    kernels = info["gpu_node_fraction"]
    if kernels is None:
        kernels_txt = "not verified"
    else:
        kernels_txt = f"{kernels * 100:.0f}% of graph nodes on GPU"

    print(bar)
    print("  Inference execution environment")
    print(bar)
    print(f"  backend            : {info['ort_package']} {info['ort_version']}".rstrip())
    print(f"  available providers: {', '.join(info['available_providers'])}")
    print(f"  requested (in order): {', '.join(info['requested_providers'])}")
    print(f"  SELECTED PROVIDER  : {info['selected_provider']}")
    print(f"  running on GPU     : {'YES' if info['on_gpu'] else 'NO  <-- CPU FALLBACK'}")
    print(f"  GPU / board        : {info['gpu_name']}")
    print(f"  JetPack (L4T)      : {info['jetpack']}")
    print(f"  CUDA / cuDNN       : {info['cuda']} / {info['cudnn']}")
    if info["tensorrt"]:
        # fp16 is None for a prebuilt engine: the precision was fixed by whoever
        # built it and is not recoverable from the serialized plan.
        fp16 = info["fp16"]
        suffix = "" if fp16 is None else f"  (fp16={'on' if fp16 else 'off'})"
        print(f"  TensorRT           : {info['tensorrt']}{suffix}")
    if info.get("engine_device_mem_mb") is not None:
        print(f"  engine device mem  : {info['engine_device_mem_mb']} MB")
    print(f"  kernel placement   : {kernels_txt}")
    if not info["on_gpu"]:
        print(bar)
        print("  " + _FIX_HINT.replace("\n", "\n  "))
    print(bar)
