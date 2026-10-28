"""Run MAX graphs on a remote Vulkan device and check them against NumPy.

The device is a phone's Adreno 710, reached through a MAX driver plugin that
forwards memory and kernel launches to a Vulkan executor on the phone
(remote/ in adityaGoCT/mojo-adreno). MAX itself, with its graph compiler and
host code, runs here; the kernels compile for `vulkan:adreno710`.

  MODULAR_DRIVER_PLUGINS="vulkan@/path/to/libM_driver_vulkan_remote.dylib" \\
    ./bazelw run --//:hal_device_context=true //max/examples/vulkan_coverage:remote
"""

import math
import sys
import time

import numpy as np
from max import driver
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

rng = np.random.default_rng(0)
GPU = DeviceRef.GPU()


def erf(x):
    return np.vectorize(math.erf)(x)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def case(name, shapes, fn, reference=None, tol=1e-4):
    """A one-graph test: `fn(device, *inputs)` builds the output on `device`.
    `reference(*arrays)` gives the expected result; without one, the same
    graph run by MAX on this machine's CPU does."""
    return name, shapes, fn, reference, tol


GAMMA = rng.standard_normal(256).astype(np.float32)
BETA = rng.standard_normal(256).astype(np.float32)
INDICES = np.array([3, 1, 4, 1, 5], np.int64)
CONV = (rng.standard_normal((3, 3, 64, 64)) * 0.05).astype(np.float32)


def const(array, dtype, device):
    return ops.constant(array, dtype, device=device)


def resnet18(device, x):
    import resnet18_graph  # noqa: PLC0415  (random weights, fixed seed)

    return resnet18_graph.forward(x, device)


CASES = [
    case("add", [(1024,), (1024,)], lambda d, a, b: a + b, lambda a, b: a + b),
    case("mul", [(1024,), (1024,)], lambda d, a, b: a * b, lambda a, b: a * b),
    case("relu", [(1024,)], lambda d, x: ops.relu(x), lambda x: np.maximum(x, 0)),
    case("exp", [(1024,)], lambda d, x: ops.exp(x), np.exp),
    case("tanh", [(1024,)], lambda d, x: ops.tanh(x), np.tanh),
    case("gelu", [(1024,)], lambda d, x: ops.gelu(x), lambda x: 0.5 * x * (1 + erf(x / math.sqrt(2)))),
    case("sigmoid", [(1024,)], lambda d, x: ops.sigmoid(x), lambda x: 1 / (1 + np.exp(-x))),
    case("silu", [(1024,)], lambda d, x: ops.silu(x), lambda x: x / (1 + np.exp(-x))),
    case("cast f32->f16", [(1024,)], lambda d, x: ops.cast(x, DType.float16), lambda x: x.astype(np.float16), 1e-3),
    case("transpose", [(64, 128)], lambda d, x: ops.transpose(x, 0, 1), lambda x: x.T),
    case("concat", [(64, 32), (64, 32)], lambda d, a, b: ops.concat([a, b], axis=1),
         lambda a, b: np.concatenate([a, b], axis=1)),
    case("slice", [(64, 128)], lambda d, x: x[:, 32:96], lambda x: x[:, 32:96]),
    case("gather", [(512, 64)], lambda d, x: ops.gather(x, const(INDICES, DType.int64, d), axis=0),
         lambda x: x[INDICES]),
    case("sum", [(64, 256)], lambda d, x: ops.sum(x, axis=1), lambda x: x.sum(axis=1, keepdims=True), 1e-3),
    case("mean", [(64, 256)], lambda d, x: ops.mean(x, axis=1), lambda x: x.mean(axis=1, keepdims=True)),
    case("argmax", [(8, 1000)], lambda d, x: ops.argmax(x, axis=1), lambda x: x.argmax(axis=1)[:, None], 0),
    case("softmax", [(64, 256)], lambda d, x: ops.softmax(x), softmax),
    case("layer_norm", [(64, 256)],
         lambda d, x: ops.layer_norm(x, const(GAMMA, DType.float32, d), const(BETA, DType.float32, d), epsilon=1e-5),
         lambda x: layer_norm(x, GAMMA, BETA), 1e-3),
    case("matmul 256x256", [(256, 256), (256, 256)], lambda d, a, b: a @ b, lambda a, b: a @ b, 1e-3),
    case("matmul f16 256x256", [(256, 256), (256, 256)],
         lambda d, a, b: ops.cast(ops.cast(a, DType.float16) @ ops.cast(b, DType.float16), DType.float32),
         lambda a, b: (a.astype(np.float16).astype(np.float32) @ b.astype(np.float16).astype(np.float32))
         .astype(np.float16), 5e-3),  # MAX's CPU device has no f16
    case("gemv 1x2048x2048", [(1, 2048), (2048, 2048)], lambda d, a, b: a @ b, lambda a, b: a @ b, 1e-3),
    case("conv2d 3x3", [(1, 56, 56, 64)],
         lambda d, x: ops.conv2d(x, const(CONV, DType.float32, d), stride=(1, 1), padding=(1, 1, 1, 1)), None, 1e-3),
    case("max_pool2d", [(1, 112, 112, 64)],
         lambda d, x: ops.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1), None, 0),
    case("avg_pool2d", [(1, 56, 56, 64)],
         lambda d, x: ops.avg_pool2d(x, kernel_size=(2, 2), stride=2), None, 1e-5),
    case("ResNet-18", [(1, 224, 224, 3)], resnet18, None, 2e-3),
]


def build(name, shapes, fn, device):
    types = [TensorType(DType.float32, s, device=device) for s in shapes]
    with Graph(name.replace(" ", "_").replace(">", "").replace("-", "_"), input_types=types) as g:
        g.output(fn(device, *[x.tensor for x in g.inputs]))
    return g


def execute(session, device, graph, inputs):
    """Runs `graph` on `device`; returns the output and the run's seconds (inputs
    in, execute, output back; not compiling)."""
    model = session.load(graph)
    t = time.time()
    out = model.execute(*[driver.Buffer.from_numpy(x).to(device) for x in inputs])[0]
    return out.to(driver.CPU()).to_numpy(), time.time() - t


def run(sessions, name, shapes, fn, reference, tol):
    (gpu_session, gpu), (cpu_session, cpu) = sessions
    inputs = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    got, seconds = execute(gpu_session, gpu, build(name, shapes, fn, GPU), inputs)
    if reference is not None:
        want = np.asarray(reference(*inputs))
    else:
        want = execute(cpu_session, cpu, build(name, shapes, fn, DeviceRef.CPU()), inputs)[0]
    want = want.reshape(got.shape)
    err = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
    scale = max(1.0, float(np.max(np.abs(want.astype(np.float64)))))
    return err <= tol * scale, err, seconds


def main():
    device = driver.Accelerator()
    print(f"device: {device}, api {driver.accelerator_api()}, arch {driver.accelerator_architecture_name()}")
    cpu = driver.CPU()
    sessions = ((InferenceSession(devices=[device]), device), (InferenceSession(devices=[cpu]), cpu))
    verbose = "--verbose" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    passed = total = 0
    for name, shapes, fn, reference, tol in CASES:
        if names and name not in names:
            continue
        total += 1
        try:
            ok, err, seconds = run(sessions, name, shapes, fn, reference, tol)
            ref = "NumPy" if reference is not None else "MAX on CPU"
            print(f"{name:20s} {'PASS' if ok else 'FAIL'}  max error {err:.2e} (vs {ref})  {seconds:.2f} s",
                  flush=True)
            passed += ok
        except Exception as e:  # report and go on
            while e.__cause__ is not None:
                e = e.__cause__
            if verbose:
                print(str(e), file=sys.stderr)
            lines = [l for l in str(e).strip().splitlines() if "source-tracebacks" not in l]
            print(f"{name:20s} ERROR {(lines[-1] if lines else type(e).__name__)[:200]}", flush=True)
    print(f"\n{passed}/{total} passed")


if __name__ == "__main__":
    main()
