# ===----------------------------------------------------------------------=== #
# Compile-only test of MAX graphs for the Vulkan SPIR-V target (prototype).
# ===----------------------------------------------------------------------=== #
"""ResNet-18 built with MAX's graph API: random weights, batch norm folded into the
convolution biases, NHWC activations and RSCF filters."""
import numpy as np
from max.dtype import DType
from max.graph import Graph, TensorType, ops
rng = np.random.default_rng(0)
W = {}  # name -> (RSCF filter, bias)

def conv_w(name, k, cin, cout):
    W[name] = ((rng.standard_normal((k, k, cin, cout)) * np.sqrt(2.0 / (k * k * cin))).astype(np.float32),
               (rng.standard_normal(cout) * 0.01).astype(np.float32))

conv_w("stem", 7, 3, 64)
blocks = []
cin = 64
for stage, cout in enumerate((64, 128, 256, 512)):
    for b in range(2):
        stride = 2 if (stage > 0 and b == 0) else 1
        n = f"s{stage}b{b}"
        conv_w(n + "c1", 3, cin, cout); conv_w(n + "c2", 3, cout, cout)
        if stride != 1 or cin != cout:
            conv_w(n + "ds", 1, cin, cout)
        blocks.append((n, cin, cout, stride))
        cin = cout
FC = ((rng.standard_normal((512, 1000)) * 0.02).astype(np.float32), np.zeros(1000, np.float32))

def build(device):
    with Graph("resnet18", input_types=[TensorType(DType.float32, (1, 224, 224, 3), device=device)]) as g:
        def conv(x, name, stride, pad, relu):
            f, b = W[name]
            y = ops.conv2d(x, ops.constant(f, DType.float32, device=device), stride=(stride, stride),
                           padding=(pad, pad, pad, pad), bias=ops.constant(b, DType.float32, device=device))
            return ops.relu(y) if relu else y
        x = conv(g.inputs[0].tensor, "stem", 2, 3, True)
        x = ops.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)
        for n, ci, co, s in blocks:
            y = conv(x, n + "c1", s, 1, True)
            y = conv(y, n + "c2", 1, 1, False)
            sc = conv(x, n + "ds", s, 0, False) if (n + "ds") in W else x
            x = ops.relu(y + sc)
        x = ops.mean(ops.mean(x, axis=1), axis=2)            # global average pool -> (1, 1, 1, 512)
        x = ops.reshape(x, (1, 512))
        x = x @ ops.constant(FC[0], DType.float32, device=device) + ops.constant(FC[1], DType.float32, device=device)
        g.output(x)
    return g

