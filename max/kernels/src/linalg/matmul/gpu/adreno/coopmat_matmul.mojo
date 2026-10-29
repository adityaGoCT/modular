# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Matmul with KHR cooperative matrices for Adreno 8xx through Vulkan SPIR-V.

One subgroup computes each 64x64 tile of C: it loads tiles of A (64xK) and B
(Kx64) as cooperative matrices, multiply-adds them into an accumulator along K,
and stores it. The driver picks how each matrix is spread across the
subgroup's threads.

The shapes are the Adreno 840's (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
on a Galaxy S26 Ultra): M is 64, N is 16, 32 or 64, and A, B and the
accumulator share one type, fp16 with K = 16 or fp32 with K = 8. There is no
fp16 x fp16 -> fp32, so fp16 products accumulate in fp16.
"""

from std.math import ceildiv

from max.gpu import block_idx
from max.gpu.compute.arch.mma_spirv import (
    ACCUMULATOR,
    MATRIX_A,
    MATRIX_B,
    CoopMatrix,
)
from max.gpu.host import DeviceContext
from layout import TensorLayout, TensorEngine, TileTensor

comptime COOPMAT_M = 64
"""Rows of the tile of C each subgroup computes (the only M the 840 offers)."""
comptime COOPMAT_N = 64
"""Columns of the tile of C each subgroup computes: of the 840's 16, 32 and 64,
the fastest (2.1 TFLOP/s in fp16 at 2048x2048x2048 against 0.65 with 16)."""
comptime COOPMAT_SUBGROUP = 64
"""Threads per workgroup: one Adreno subgroup."""


def coopmat_k[dtype: DType]() -> Int:
    """Returns the K of one cooperative-matrix product for `dtype`.

    Parameters:
        dtype: The element type of A, B and C: float16 or float32.

    Returns:
        16 for float16 and 8 for float32.
    """
    comptime if dtype == DType.float16:
        return 16
    else:
        comptime assert dtype == DType.float32, "float16 or float32 only"
        return 8


def adreno_coopmat_supports[
    c_type: DType, a_type: DType, b_type: DType
]() -> Bool:
    """Returns whether `matmul_kernel_adreno_coopmat` takes these types.

    Parameters:
        c_type: The element type of C.
        a_type: The element type of A.
        b_type: The element type of B.

    Returns:
        True when A, B and C are all float16 or all float32.
    """
    comptime float_types = (DType.float16, DType.float32)
    return a_type == b_type and a_type == c_type and a_type in float_types


def adreno_coopmat_fits[dtype: DType](m: Int, n: Int, k: Int) -> Bool:
    """Returns whether `matmul_kernel_adreno_coopmat` covers an M x N x K
    product exactly.

    Parameters:
        dtype: The element type of A, B and C.

    Args:
        m: Rows of C.
        n: Columns of C.
        k: The reduced dimension.

    Returns:
        True when M, N and K are multiples of the tile shape.
    """
    return (
        m % COOPMAT_M == 0
        and n % COOPMAT_N == 0
        and k % coopmat_k[dtype]() == 0
    )


@__name(
    t"matmul_kernel_adreno_coopmat_{c_type}_{a_type}_{b_type}_{transpose_b}",
)
def matmul_kernel_adreno_coopmat[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    transpose_b: Bool = False,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
    m: Int32,
    n: Int32,
    k: Int32,
):
    """Computes the 64x64 tile of `c = a x b` at block (x: column tile, y: row
    tile) with one subgroup, accumulating in C's type.

    Requires `adreno_coopmat_supports` types, `adreno_coopmat_fits` sizes and
    unit column strides; rows may be strided. With `transpose_b`, `b` is stored
    as (N, K).
    """
    comptime assert c.flat_rank == 2, "c must have flat_rank == 2"
    comptime assert a.flat_rank == 2, "a must have flat_rank == 2"
    comptime assert b.flat_rank == 2, "b must have flat_rank == 2"
    comptime assert adreno_coopmat_supports[
        c_type, a_type, b_type
    ](), "A, B and C must all be float16 or all float32"
    comptime M = COOPMAT_M
    comptime N = COOPMAT_N
    comptime K = coopmat_k[c_type]()

    var row = Int(block_idx.y) * M
    var col = Int(block_idx.x) * N
    var lda = Int(a.layout.stride[0]().value())
    var ldb = Int(b.layout.stride[0]().value())
    var ldc = Int(c.layout.stride[0]().value())

    var acc = CoopMatrix[c_type, M, N, ACCUMULATOR].zero()
    for kk in range(0, Int(k), K):
        var a_tile = CoopMatrix[a_type, M, K, MATRIX_A].load(
            a.ptr.unsafe_offset(row * lda + kk), lda
        )
        comptime if transpose_b:
            # The (K, N) tile of an (N, K) buffer is column-major.
            var b_tile = CoopMatrix[b_type, K, N, MATRIX_B].load[
                row_major=False
            ](b.ptr.unsafe_offset(col * ldb + kk), ldb)
            acc = a_tile.multiply_add(b_tile, acc)
        else:
            var b_tile = CoopMatrix[b_type, K, N, MATRIX_B].load(
                b.ptr.unsafe_offset(kk * ldb + col), ldb
            )
            acc = a_tile.multiply_add(b_tile, acc)
    acc.store(c.ptr.unsafe_offset(row * ldc + col), ldc)


def enqueue_adreno_coopmat_matmul[
    *,
    transpose_b: Bool,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[mut=False, ...],
    b: TileTensor[mut=False, ...],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Enqueues `matmul_kernel_adreno_coopmat`, one subgroup per 64x64 tile of
    `c`.

    Parameters:
        transpose_b: Whether `b` is stored as (N, K).

    Args:
        c: Rank-2 output tensor (M, N).
        a: Rank-2 left operand (M, K).
        b: Rank-2 right operand (K, N), or (N, K) with `transpose_b`.
        m: Rows of `c`, a multiple of 64.
        n: Columns of `c`, a multiple of 64.
        k: The reduced dimension, a multiple of 16 (float16) or 8 (float32).
        ctx: Device context for the launch.
    """
    comptime kernel = matmul_kernel_adreno_coopmat[
        c.dtype,
        a.dtype,
        b.dtype,
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(b).Engine,
        transpose_b,
    ]
    ctx.enqueue_function[kernel](
        c,
        a,
        b,
        Int32(m),
        Int32(n),
        Int32(k),
        grid_dim=(ceildiv(n, COOPMAT_N), ceildiv(m, COOPMAT_M)),
        block_dim=COOPMAT_SUBGROUP,
    )
