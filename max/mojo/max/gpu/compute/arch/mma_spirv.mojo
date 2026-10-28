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
"""Cooperative matrices (SPV_KHR_cooperative_matrix) for Vulkan SPIR-V targets.

A cooperative matrix is a tile spread across the threads of a subgroup, which
load, multiply-add and store it together; the driver picks the layout. Every
thread of the subgroup must reach each operation.

Each operation is a call to a marker function that the SPIR-V backend's kernel
ABI pass rewrites into an `OpCooperativeMatrix*KHR` instruction, carrying the
matrix in between as an opaque handle. Kernels that use them run only where the
Vulkan driver exposes `VK_KHR_cooperative_matrix` with the chosen shape and
types (query `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`); the
`adreno840` target is the one that turns them on in MAX.

Supported operations:
- Float multiply-accumulate: {F16, BF16, F32} inputs, F16 or F32 accumulator.
- Conversion between float element types (`cast`).

Example:

```mojo
from max.gpu.compute.arch.mma_spirv import CoopMatrix, MATRIX_A, MATRIX_B, ACCUMULATOR

# c[16x16] = a[16xK] x b[Kx16], one subgroup.
var acc = CoopMatrix[DType.float32, 16, 16, ACCUMULATOR].zero()
for k in range(0, K, 16):
    var a_tile = CoopMatrix[DType.float16, 16, 16, MATRIX_A].load(a.unsafe_offset(k), K)
    var b_tile = CoopMatrix[DType.float16, 16, 16, MATRIX_B].load(b.unsafe_offset(k * N), N)
    acc = a_tile.multiply_add(b_tile, acc)
acc.store(c, N)
```
"""

from std.ffi import external_call

comptime MATRIX_A = 0
"""The left operand of a multiply-add (`MatrixAKHR`)."""
comptime MATRIX_B = 1
"""The right operand of a multiply-add (`MatrixBKHR`)."""
comptime ACCUMULATOR = 2
"""The accumulator and result of a multiply-add (`MatrixAccumulatorKHR`)."""


@inline(.always)
def _dtype_code[dtype: DType]() -> Int32:
    """The element-type code of the cooperative-matrix markers."""
    comptime if dtype == DType.float16:
        return 0
    elif dtype == DType.float32:
        return 1
    elif dtype == DType.bfloat16:
        return 2
    elif dtype == DType.int8:
        return 3
    elif dtype == DType.int32:
        return 4
    else:
        comptime assert False, "unsupported cooperative matrix element type"


@inline(.always)
def _layout[row_major: Bool]() -> Int32:
    """The SPIR-V `CooperativeMatrixLayoutKHR` operand."""
    return 0 if row_major else 1


struct CoopMatrix[dtype: DType, rows: Int, cols: Int, use: Int](
    TrivialRegisterPassable
):
    """A `rows` x `cols` matrix spread across the threads of a subgroup.

    Parameters:
        dtype: Element type.
        rows: Number of rows.
        cols: Number of columns.
        use: `MATRIX_A`, `MATRIX_B` or `ACCUMULATOR`.
    """

    var _handle: UInt64

    @inline(.always)
    def __init__(out self, *, handle: UInt64):
        """Wraps a marker's handle.

        Args:
            handle: The handle a cooperative-matrix marker returned.
        """
        comptime assert (
            Self.use >= MATRIX_A and Self.use <= ACCUMULATOR
        ), "use must be MATRIX_A, MATRIX_B or ACCUMULATOR"
        self._handle = handle

    @staticmethod
    @inline(.always)
    def load[
        row_major: Bool = True
    ](ptr: Pointer[Scalar[Self.dtype], ...], stride: Int) -> Self:
        """Loads the matrix from memory, cooperatively.

        Parameters:
            row_major: Whether rows (True) or columns (False) are contiguous.

        Args:
            ptr: Pointer to the first element of the matrix. It must point
                into a kernel buffer or shared memory.
            stride: Elements between consecutive rows (or columns).

        Returns:
            The loaded matrix.
        """
        return Self(
            handle=external_call["__mojo_spirv_coopmat_load", UInt64](
                ptr,
                Int32(stride),
                _layout[row_major](),
                _dtype_code[Self.dtype](),
                Int32(Self.rows),
                Int32(Self.cols),
                Int32(Self.use),
            )
        )

    @staticmethod
    @inline(.always)
    def zero() -> Self:
        """Returns the matrix with every element zero.

        Returns:
            A zero matrix.
        """
        return Self(
            handle=external_call["__mojo_spirv_coopmat_zero", UInt64](
                _dtype_code[Self.dtype](),
                Int32(Self.rows),
                Int32(Self.cols),
                Int32(Self.use),
            )
        )

    @inline(.always)
    def multiply_add[
        b_dtype: DType,
        c_dtype: DType,
        n: Int,
    ](
        self,
        b: CoopMatrix[b_dtype, Self.cols, n, MATRIX_B],
        c: CoopMatrix[c_dtype, Self.rows, n, ACCUMULATOR],
    ) -> CoopMatrix[c_dtype, Self.rows, n, ACCUMULATOR]:
        """Returns `c + self x b`, with this matrix as the left operand.

        Parameters:
            b_dtype: Element type of `b`.
            c_dtype: Element type of the accumulator.
            n: Columns of `b` and the accumulator.

        Args:
            b: The right operand.
            c: The accumulator.

        Returns:
            The sum `c + self x b`.
        """
        comptime assert (
            Self.use == MATRIX_A
        ), "the left operand must be MATRIX_A"
        comptime assert (
            c_dtype.is_floating_point()
        ), "integer cooperative matrix products are not supported yet"
        return CoopMatrix[c_dtype, Self.rows, n, ACCUMULATOR](
            handle=external_call["__mojo_spirv_coopmat_mma", UInt64](
                self._handle, b._handle, c._handle
            )
        )

    @inline(.always)
    def cast[
        target: DType
    ](self) -> CoopMatrix[target, Self.rows, Self.cols, Self.use]:
        """Converts every element to another float type.

        Parameters:
            target: The new element type.

        Returns:
            The converted matrix.
        """
        comptime assert (
            Self.dtype.is_floating_point()
        ), "only float cooperative matrices can be converted"
        comptime assert (
            target.is_floating_point()
        ), "only float cooperative matrices can be converted"
        comptime if target == Self.dtype:
            return CoopMatrix[target, Self.rows, Self.cols, Self.use](
                handle=self._handle
            )
        return CoopMatrix[target, Self.rows, Self.cols, Self.use](
            handle=external_call["__mojo_spirv_coopmat_convert", UInt64](
                self._handle, _dtype_code[target]()
            )
        )

    @inline(.always)
    def store[
        row_major: Bool = True
    ](self, ptr: Pointer[mut=True, Scalar[Self.dtype], ...], stride: Int):
        """Stores the matrix to memory, cooperatively.

        Parameters:
            row_major: Whether rows (True) or columns (False) are contiguous.

        Args:
            ptr: Pointer to the first element of the matrix. It must point
                into a kernel buffer or shared memory.
            stride: Elements between consecutive rows (or columns).
        """
        external_call["__mojo_spirv_coopmat_store", NoneType](
            ptr,
            self._handle,
            Int32(stride),
            _layout[row_major](),
            _dtype_code[Self.dtype](),
        )
