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

One subgroup computes each 16x16 tile of C: it loads 16x16 tiles of A and B as
cooperative matrices, multiply-adds them into an f32 accumulator along K, and
stores the accumulator converted to C's type. The driver picks how each matrix
is spread across the subgroup's threads.
"""

from std.math import ceildiv
from std.utils.numerics import get_accum_type

from max.gpu import block_idx
from max.gpu.compute.arch.mma_spirv import (
    ACCUMULATOR,
    MATRIX_A,
    MATRIX_B,
    CoopMatrix,
)
from max.gpu.host import DeviceContext
from layout import TensorLayout, TensorEngine, TileTensor

comptime COOPMAT_TILE = 16
"""Rows and columns of every cooperative-matrix tile (16x16x16 products)."""
comptime COOPMAT_SUBGROUP = 64
"""Threads per workgroup: one Adreno subgroup."""


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
    """Computes the 16x16 tile of `c = a x b` at block (x: column tile, y: row
    tile) with one subgroup.

    Requires M, N and K to be multiples of 16 and unit column strides; rows may
    be strided. With `transpose_b`, `b` is stored as (N, K).
    """
    comptime assert c.flat_rank == 2, "c must have flat_rank == 2"
    comptime assert a.flat_rank == 2, "a must have flat_rank == 2"
    comptime assert b.flat_rank == 2, "b must have flat_rank == 2"
    comptime TILE = COOPMAT_TILE
    comptime s_type = get_accum_type[c_type]()

    var row = Int(block_idx.y) * TILE
    var col = Int(block_idx.x) * TILE
    var lda = Int(a.layout.stride[0]().value())
    var ldb = Int(b.layout.stride[0]().value())
    var ldc = Int(c.layout.stride[0]().value())

    var acc = CoopMatrix[s_type, TILE, TILE, ACCUMULATOR].zero()
    for kk in range(0, Int(k), TILE):
        var a_tile = CoopMatrix[a_type, TILE, TILE, MATRIX_A].load(
            a.ptr.unsafe_offset(row * lda + kk), lda
        )
        comptime if transpose_b:
            # The (K, N) tile of an (N, K) buffer is column-major.
            var b_tile = CoopMatrix[b_type, TILE, TILE, MATRIX_B].load[
                row_major=False
            ](b.ptr.unsafe_offset(col * ldb + kk), ldb)
            acc = a_tile.multiply_add(b_tile, acc)
        else:
            var b_tile = CoopMatrix[b_type, TILE, TILE, MATRIX_B].load(
                b.ptr.unsafe_offset(kk * ldb + col), ldb
            )
            acc = a_tile.multiply_add(b_tile, acc)
    acc.cast[c_type]().store(c.ptr.unsafe_offset(row * ldc + col), ldc)


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
    """Enqueues `matmul_kernel_adreno_coopmat`, one subgroup per 16x16 tile of
    `c`.

    Parameters:
        transpose_b: Whether `b` is stored as (N, K).

    Args:
        c: Rank-2 output tensor (M, N).
        a: Rank-2 left operand (M, K).
        b: Rank-2 right operand (K, N), or (N, K) with `transpose_b`.
        m: Rows of `c`, a multiple of 16.
        n: Columns of `c`, a multiple of 16.
        k: The reduced dimension, a multiple of 16.
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
        grid_dim=(ceildiv(n, COOPMAT_TILE), ceildiv(m, COOPMAT_TILE)),
        block_dim=COOPMAT_SUBGROUP,
    )
