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
"""Standard library hooks for Vulkan compute (logical SPIR-V) targets.

GPU indices read the SPIR-V builtins through LLVM's `llvm.spv.*` intrinsics.
A Mojo warp is a 64-lane segment of the Vulkan subgroup, which is 64 lanes on
Qualcomm's Adreno driver and 128 on Mesa Turnip, so `lane_id` is the subgroup
lane modulo 64.
"""

from std._gpu.primitives.id import _GPUIdPluginHookFnType, _is_cluster_id
from std._plugin._trait import (
    TargetDeviceKind,
    PluginHooks,
    StdPlugin,
    _PrintEmitPluginHookFnType,
    _UnaryMathPluginHookFnType,
)
from std.math.math import _UnaryMathOp
from std.atomic import Ordering
from std.collections import OptionalReg
from std.collections.string.string_span import _get_kgen_string
from std.ffi import CStringSpan
from std.io.file_descriptor import RawFileDescriptor
from std.sys import llvm_intrinsic
from std.sys.info import CompilationTarget
from std.sys.intrinsics import PrefetchOptions, _PrefetchPluginHookFnType


def _drop_print[O: Origin](cstr: CStringSpan[O], file: RawFileDescriptor):
    """Drops `print()` output from a kernel.

    Vulkan shaders have no `printf` (only `debugPrintfEXT`, under the
    validation layers), so device prints, including the messages of
    `debug_assert`s compiled into kernels, are discarded.
    """
    pass


@inline(.always)
def _drop_prefetch[
    address_space: AddressSpace
](addr: Pointer[NoneType, ImmUntrackedOrigin, address_space=address_space]):
    # SPIR-V has no prefetch instruction, and dropping a hint keeps the program
    # correct.
    pass


@inline(.always)
def _dim_index[dim: StaticString]() -> Int32:
    comptime if dim == "x":
        return 0
    elif dim == "y":
        return 1
    else:
        comptime assert dim == "z", "unknown GPU dimension"
        return 2


@inline(.always)
def _spv_read[intrinsic: StaticString, dim: StaticString]() -> Int:
    return Int(
        llvm_intrinsic[intrinsic, UInt32, has_side_effect=False](
            _dim_index[dim]()
        )
    )


def _spv_gpu_id_fn[
    name: StaticString, dim: StaticString
]() -> OptionalReg[_GPUIdPluginHookFnType]:
    """Returns the SPIR-V read of `name` along `dim`, or `None` for a thread
    block cluster id, which Vulkan does not have."""
    comptime if _is_cluster_id(name):
        return None
    else:
        return comptime (
            OptionalReg[_GPUIdPluginHookFnType](_spv_gpu_id[name, dim])
        )


@inline(.always)
def _spv_gpu_id[name: StaticString, dim: StaticString]() -> Int:
    """Reads the GPU index or size `name` along `dim` from its SPIR-V builtin.
    """
    comptime if name == "lane_id":
        return Int(
            llvm_intrinsic[
                "llvm.spv.subgroup.local.invocation.id",
                UInt32,
                has_side_effect=False,
            ]()
            & 63
        )
    elif name == "thread_idx":
        return _spv_read["llvm.spv.thread.id.in.group", dim]()
    elif name == "block_idx":
        return _spv_read["llvm.spv.group.id", dim]()
    elif name == "block_dim":
        # Vulkan fixes the workgroup size in the shader; the backend folds this
        # read into that constant.
        return _spv_read["llvm.spv.workgroup.size", dim]()
    else:
        comptime assert name == "grid_dim", "unknown GPU id"
        return _spv_read["llvm.spv.num.workgroups", dim]()


comptime _MathFn = OptionalReg[_UnaryMathPluginHookFnType]


def _spv_unary_math_fn[op: _UnaryMathOp, dtype: DType]() -> _MathFn:
    """Returns the lowering of the `std.math` function `op` to LLVM's math
    intrinsic, which the SPIR-V backend emits as the GLSL.std.450 instruction
    (Exp2, Sqrt, InverseSqrt, ...), or `None` to keep the stdlib's own
    lowering. GLSL.std.450 has 16- and 32-bit float overloads only."""
    comptime if dtype not in (DType.float16, DType.float32):
        return None
    comptime match op:
        case .exp2:
            return comptime (_MathFn(_spv_unary_math[name="llvm.exp2"]))
        case .exp:
            return comptime (_MathFn(_spv_unary_math[name="llvm.exp"]))
        case .log:
            return comptime (_MathFn(_spv_unary_math[name="llvm.log"]))
        case .log2:
            return comptime (_MathFn(_spv_unary_math[name="llvm.log2"]))
        case .sqrt:
            return comptime (_MathFn(_spv_unary_math[name="llvm.sqrt"]))
        case .rsqrt:
            return comptime (_MathFn(_spv_unary_math[name="llvm.spv.rsqrt"]))
        case .sin:
            return comptime (_MathFn(_spv_unary_math[name="llvm.sin"]))
        case .cos:
            return comptime (_MathFn(_spv_unary_math[name="llvm.cos"]))
        case .recip | .tanh:
            return None


def _spv_unary_math[
    dtype: DType, width: SIMDLength, //, name: StaticString
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Lowers a `std.math` function to the LLVM intrinsic `name`."""
    return llvm_intrinsic[name, type_of(x), has_side_effect=False](x)


struct SPIRVPlugin(StdPlugin):
    """`PluginHooks` implementation for Vulkan compute (SPIR-V) backends."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["spirv"]()
    comptime HookFor[target: CompilationTarget] = SPIRVPluginHooks[target]


struct SPIRVPluginHooks[target: CompilationTarget](PluginHooks):
    comptime device_kind = TargetDeviceKind.gpu

    comptime gpu_id_fn[name: StaticString, dim: StaticString]: OptionalReg[
        _GPUIdPluginHookFnType
    ] = _spv_gpu_id_fn[name, dim]()

    comptime print_emit_fn: OptionalReg[
        _PrintEmitPluginHookFnType
    ] = _drop_print

    comptime unary_math_fn[
        op: _UnaryMathOp, dtype: DType, width: Int
    ]: OptionalReg[_UnaryMathPluginHookFnType] = _spv_unary_math_fn[
        op, dtype
    ]()

    comptime supports_float64: Bool = False

    # Logical SPIR-V has no generic address space: a pointer rebuilt from an
    # integer cannot reach a buffer or Workgroup memory.
    comptime has_generic_address_space: Bool = False

    comptime prefetch_fn[
        address_space: AddressSpace, params: PrefetchOptions
    ]: OptionalReg[_PrefetchPluginHookFnType[address_space]] = _drop_prefetch[
        address_space
    ]

    comptime default_atomic_ordering: Ordering = Ordering.RELAXED
