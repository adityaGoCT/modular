"""Compile-only test of MAX graphs for the Vulkan SPIR-V targets (Adreno).

MAX's virtual-device mode stands in for a GPU that is not present: the graph
compiler and MAX's kernel library compile for `vulkan:adreno710` (or the arch
given with `--arch=`) exactly as for a real device, and nothing runs. Each case
is a one-op graph, then a full ResNet-18; the report says which MAX kernels
compile for the target. `adreno840` takes the cooperative-matrix matmul.

  ./bazelw run //max/examples/vulkan_coverage:coverage [-- --arch=adreno840]
"""

import re
import subprocess
import sys
import time

import numpy as np
from max import driver
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

driver.set_virtual_device_api("vulkan")
ARCH = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--arch=")), "adreno710")
driver.set_virtual_device_target_arch(ARCH)
driver.set_virtual_device_count(1)

GPU = DeviceRef.GPU()
rng = np.random.default_rng(0)


def const(shape, dtype=DType.float32):
    return ops.constant(rng.standard_normal(shape).astype(np.float32), dtype, device=GPU)


def one_input(name, shape, fn, dtype=DType.float32):
    def build():
        with Graph(name, input_types=[TensorType(dtype, shape, device=GPU)]) as g:
            g.output(fn(g.inputs[0].tensor))
        return g
    return name, build


def two_inputs(name, shape_a, shape_b, fn, dtype=DType.float32):
    def build():
        with Graph(name, input_types=[TensorType(dtype, shape_a, device=GPU),
                                      TensorType(dtype, shape_b, device=GPU)]) as g:
            g.output(fn(g.inputs[0].tensor, g.inputs[1].tensor))
        return g
    return name, build


def resnet18():
    import resnet18_graph  # noqa: PLC0415  (builds its random weights on import)

    return resnet18_graph.build(GPU)


CASES = [
    two_inputs("add", (1024,), (1024,), lambda a, b: a + b),
    two_inputs("mul", (1024,), (1024,), lambda a, b: a * b),
    one_input("relu", (1024,), ops.relu),
    one_input("exp", (1024,), ops.exp),
    one_input("tanh", (1024,), ops.tanh),
    one_input("gelu", (1024,), ops.gelu),
    one_input("sigmoid", (1024,), ops.sigmoid),
    one_input("silu", (1024,), ops.silu),
    one_input("cast f32->f16", (1024,), lambda x: ops.cast(x, DType.float16)),
    one_input("transpose", (64, 128), lambda x: ops.transpose(x, 0, 1)),
    two_inputs("concat", (64, 32), (64, 32), lambda a, b: ops.concat([a, b], axis=1)),
    one_input("slice", (64, 128), lambda x: x[:, 32:96]),
    one_input("gather", (512, 64), lambda x: ops.gather(
        x, ops.constant(np.array([3, 1, 4, 1, 5], np.int64), DType.int64, device=GPU), axis=0)),
    one_input("sum (reduce)", (64, 256), lambda x: ops.sum(x, axis=1)),
    one_input("mean (reduce)", (64, 256), lambda x: ops.mean(x, axis=1)),
    one_input("argmax", (8, 1000), lambda x: ops.argmax(x, axis=1)),
    one_input("softmax", (64, 256), ops.softmax),
    one_input("layer_norm", (64, 256), lambda x: ops.layer_norm(
        x, const((256,)), const((256,)), epsilon=1e-5)),
    two_inputs("matmul f32 256x256", (256, 256), (256, 256), lambda a, b: a @ b),
    two_inputs("matmul f16 256x256", (256, 256), (256, 256), lambda a, b: a @ b, dtype=DType.float16),
    two_inputs("matmul f16 a @ b.T", (256, 128), (64, 128), lambda a, b: a @ b.T, dtype=DType.float16),
    two_inputs("matmul (gemv) 1x2048x2048", (1, 2048), (2048, 2048), lambda a, b: a @ b),
    one_input("conv2d 3x3", (1, 56, 56, 64), lambda x: ops.conv2d(
        x, const((3, 3, 64, 64)), stride=(1, 1), padding=(1, 1, 1, 1))),
    one_input("max_pool2d", (1, 112, 112, 64), lambda x: ops.max_pool2d(
        x, kernel_size=(3, 3), stride=2, padding=1)),
    one_input("avg_pool2d", (1, 56, 56, 64), lambda x: ops.avg_pool2d(
        x, kernel_size=(2, 2), stride=2)),
    ("ResNet-18 (full graph)", resnet18),
]


def first_error(e):
    while e.__cause__ is not None:  # MAX wraps the compiler's error in a generic one
        e = e.__cause__
    text = str(e)
    lines = [l.strip() for l in text.splitlines()
             if re.search(r"error|unsupported|failed|constraint", l, re.I) and "Failed to compile the model" not in l]
    return (lines[-1] if lines else text.strip().splitlines()[0] if text.strip() else type(e).__name__)[:260]


def run_case(name):
    """Compiles one case; prints `RESULT|status|detail` for the parent."""
    build = dict(CASES)[name]
    try:
        graph = build()
    except Exception as e:  # graph API, not the target
        print(f"RESULT|API|{first_error(e)}", flush=True)
        return
    session = InferenceSession(devices=[driver.Accelerator()])
    t = time.time()
    try:
        session.load(graph)
        print(f"RESULT|PASS|{time.time() - t:.1f} s", flush=True)
    except Exception as e:
        if "--verbose" in sys.argv:
            cause = e
            while cause.__cause__ is not None:
                cause = cause.__cause__
            print(str(cause), file=sys.stderr)
        print(f"RESULT|FAIL|{first_error(e)}", flush=True)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--case":
        run_case(sys.argv[2])
        return
    print(f"device: {driver.Accelerator()}  (virtual, compile-only)")
    results = []
    for name, _ in CASES:  # one process per case: a compiler abort ends only that case
        r = subprocess.run([sys.executable, sys.argv[0], "--case", name, f"--arch={ARCH}"],
                           capture_output=True, text=True, timeout=1800)
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT|")), None)
        if line:
            _, status, detail = line.split("|", 2)
        else:
            crash = [l.strip() for l in r.stderr.splitlines() if re.search(r"UNREACHABLE|error|Assertion", l)]
            status, detail = "CRASH", (crash[0] if crash else f"exit {r.returncode}")[:260]
        results.append((name, status, detail))
        print(f"{name:28s} {status:5s} {detail}", flush=True)
    passed = sum(r[1] == "PASS" for r in results)
    print(f"\n{passed}/{len(results)} compile for vulkan:{ARCH}")


if __name__ == "__main__":
    main()
