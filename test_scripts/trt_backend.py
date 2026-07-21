"""Native TensorRT inference backend for a prebuilt ``.engine`` file.

Why this exists
---------------
ONNX Runtime's TensorRT execution provider only loads engines *it* built, keyed
by an internal subgraph hash (``TensorrtExecutionProvider_TRTKernel_graph_...``).
A standalone engine produced by ``trtexec`` or the TensorRT API — like
``ffc_rs_26.engine`` — is invisible to it, and asking ORT for TensorRT instead
triggers an ~8 minute on-device engine build that the 8 GB Orin Nano struggles
with. Loading the engine directly skips both problems and drops the whole ORT
graph, session and CUDA arena from the process.

The class deliberately mimics the small slice of ``onnxruntime.InferenceSession``
that the detector uses (``run``, ``get_inputs``, ``get_providers``), so
``object_detection.detect()`` works against either backend unchanged.

Memory / copy behaviour
-----------------------
Everything is allocated once at construction and reused for the life of the
process: device buffers for every I/O tensor, page-locked host staging buffers,
and one CUDA stream. A steady-state frame performs exactly one H2D copy, one
``execute_async_v3``, and one D2H copy per output — no allocation, no
synchronisation beyond a single stream sync.

Pinned (page-locked) host memory matters here: pageable memory forces the driver
to stage transfers through an internal bounce buffer, roughly halving effective
PCIe/iGPU copy bandwidth.
"""

import os

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

_TRT_TO_NUMPY = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT32: np.int32,
    trt.DataType.INT64: np.int64,
    trt.DataType.INT8: np.int8,
    trt.DataType.BOOL: np.bool_,
}


def _check(err, *rest):
    """Unwrap a cuda-python ``(err, value...)`` return, raising on failure."""
    if isinstance(err, cudart.cudaError_t) and err != cudart.cudaError_t.cudaSuccess:
        name = cudart.cudaGetErrorString(err)[1].decode()
        raise RuntimeError(f"CUDA error: {name}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else rest


class _Binding:
    """One engine I/O tensor: device buffer + pinned host view of the same shape."""

    __slots__ = ("name", "shape", "dtype", "nbytes", "device", "host", "is_input")

    def __init__(self, name, shape, dtype, is_input):
        self.name = name
        self.shape = tuple(shape)
        self.dtype = dtype
        self.is_input = is_input
        self.nbytes = int(np.prod(self.shape)) * np.dtype(dtype).itemsize

        self.device = _check(*cudart.cudaMalloc(self.nbytes))
        host_ptr = _check(*cudart.cudaHostAlloc(
            self.nbytes, cudart.cudaHostAllocDefault))
        # Wrap the pinned allocation as a numpy array without copying it.
        self.host = np.ctypeslib.as_array(
            (np.ctypeslib.ctypes.c_char * self.nbytes).from_address(host_ptr)
        ).view(dtype).reshape(self.shape)

    def free(self):
        if self.device:
            cudart.cudaFree(self.device)
            self.device = None


class TensorRTSession:
    """Runs a prebuilt TensorRT engine behind an ``InferenceSession``-like API.

    Args:
        engine_path: path to a serialized ``.engine`` file.

    Raises:
        RuntimeError: if the engine cannot be deserialized (most often because it
            was built for a different GPU architecture or TensorRT version — an
            engine is not portable across either).
    """

    def __init__(self, engine_path):
        self.engine_path = engine_path
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")

        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize {engine_path}. TensorRT engines are tied to "
                "the exact GPU architecture and TensorRT version that built them; "
                "rebuild it on this board or unset TRT_ENGINE_PATH to use ONNX Runtime."
            )

        self.context = self.engine.create_execution_context()
        self.stream = _check(*cudart.cudaStreamCreate())

        self._inputs, self._outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = _TRT_TO_NUMPY[self.engine.get_tensor_dtype(name)]
            shape = self.engine.get_tensor_shape(name)
            if any(d < 0 for d in shape):
                raise RuntimeError(
                    f"{engine_path}: tensor '{name}' has a dynamic shape {tuple(shape)}. "
                    "This backend targets the detector's fixed 416x416 engine."
                )
            binding = _Binding(name, shape, dtype, is_input)
            # Addresses are constant for the session's life, so bind them once
            # rather than per inference.
            self.context.set_tensor_address(name, int(binding.device))
            (self._inputs if is_input else self._outputs).append(binding)

        if len(self._inputs) != 1:
            raise RuntimeError(f"{engine_path}: expected 1 input, got {len(self._inputs)}")

    # --- InferenceSession-compatible surface -----------------------------

    class _IOMeta:
        """Mimics ``onnxruntime.NodeArg`` for ``get_inputs()`` callers."""

        def __init__(self, binding):
            self.name = binding.name
            self.shape = list(binding.shape)
            self.type = f"tensor({np.dtype(binding.dtype).name})"

    def get_inputs(self):
        return [self._IOMeta(b) for b in self._inputs]

    def get_outputs(self):
        return [self._IOMeta(b) for b in self._outputs]

    def get_providers(self):
        return ["TensorrtExecutionProvider"]

    def run(self, output_names, feed):
        """Run one inference.

        Args:
            output_names: accepted for API compatibility; ``None`` (all outputs)
                is the only supported value, matching how the detector calls it.
            feed: ``{input_name: ndarray}``. The array is copied into pinned
                memory, so the caller may reuse its own buffer immediately.

        Returns:
            list of output arrays in engine order. **The arrays are the session's
            reusable pinned buffers** — copy them if you need to keep a result
            past the next ``run()``. The detector consumes ``output0`` within the
            same frame, so no copy is made.
        """
        inp = self._inputs[0]
        tensor = feed[inp.name]
        # Stage into pinned memory. np.copyto handles a non-contiguous or
        # differently-typed source without allocating a temporary.
        np.copyto(inp.host, tensor, casting="unsafe")

        _check(cudart.cudaMemcpyAsync(
            inp.device, inp.host.ctypes.data, inp.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))

        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("TensorRT execute_async_v3 failed")

        for out in self._outputs:
            _check(cudart.cudaMemcpyAsync(
                out.host.ctypes.data, out.device, out.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))

        _check(cudart.cudaStreamSynchronize(self.stream))
        return [out.host for out in self._outputs]

    # --- lifecycle -------------------------------------------------------

    def close(self):
        """Release device buffers, the stream, and the engine."""
        for b in self._inputs + self._outputs:
            b.free()
        if self.stream:
            cudart.cudaStreamDestroy(self.stream)
            self.stream = None
        self.context = None
        self.engine = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def describe(self):
        """Diagnostics dict, merged into the startup banner by ``runtime_info``."""
        return {
            "engine_path": self.engine_path,
            "tensorrt": trt.__version__,
            "inputs": [(b.name, b.shape, np.dtype(b.dtype).name) for b in self._inputs],
            "outputs": [(b.name, b.shape, np.dtype(b.dtype).name) for b in self._outputs],
            "device_mem_mb": round(self.engine.device_memory_size / 1e6, 1),
        }


def default_engine_path():
    """``TRT_ENGINE_PATH`` if set, else ``ffc_rs_26.engine`` beside this file."""
    return os.environ.get(
        "TRT_ENGINE_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffc_rs_26.engine"),
    )
