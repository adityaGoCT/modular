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
"""MAX hooks for Vulkan compute (logical SPIR-V) targets.

A warp is 64 lanes, a 64-lane segment of the Vulkan subgroup (64 lanes on
Qualcomm's Adreno driver, 128 on Mesa Turnip). Shuffles are subgroup shuffles
(`OpGroupNonUniformShuffle`) within that segment. A whole-warp sum is one
subgroup reduction (`OpGroupNonUniformFAdd`/`IAdd`): exact when the subgroup
is the warp, as on Qualcomm's driver, or when a workgroup has at most 64
threads. On Adreno it is several times faster than the butterfly of shuffles,
and unlike `OpGroupNonUniformShuffleXor` it is also correct after divergent
control flow, which Qualcomm's driver does not reconverge.
"""

from std.atomic import Ordering, fence
from std.collections.string.string_span import _get_kgen_string
from std.collections import OptionalReg
from std.sys import llvm_intrinsic
from std.sys.info import CompilationTarget

from max.gpu.sync.sync import (
    _BarrierPluginHookFnType,
    _SyncwarpPluginHookFnType,
)

from .._trait import MaxPlugin, MaxPluginHooks


@inline(.always)
def _barrier():
    # OpControlBarrier at workgroup scope, AcquireRelease on Workgroup memory:
    # GLSL's `barrier()`, which orders shared memory across the block. LLVM's
    # all-memory variant also orders storage buffers but at device scope, which
    # makes Adreno publish memory to the whole GPU at every barrier (2x slower
    # on a shared-memory tree reduction); SPIR-V's workgroup-scope ordering of
    # storage buffers, as in `__syncthreads`, has no LLVM intrinsic.
    llvm_intrinsic["llvm.spv.group.memory.barrier.with.group.sync", NoneType]()


@inline(.always)
def _syncwarp(mask: Int):
    # OpMemoryBarrier at subgroup scope, AcquireRelease: orders the warp’s
    # memory accesses. Vulkan has no subgroup execution barrier that LLVM can
    # emit; Adreno runs a subgroup in lockstep, as AMD waves do, where
    # `syncwarp` is only a scheduling barrier.
    fence[Ordering.ACQUIRE_RELEASE, scope="subgroup"]()


@inline(.always)
def _warp_base() -> UInt32:
    """Returns the subgroup lane of this warp's lane 0."""
    return llvm_intrinsic[
        "llvm.spv.subgroup.local.invocation.id",
        UInt32,
        has_side_effect=False,
    ]() & ~UInt32(63)


struct SPIRVMaxPlugin(MaxPlugin):
    """Provides MAX behavior for Vulkan compute (SPIR-V) targets."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["spirv"]()
    comptime HookFor[target: CompilationTarget] = SPIRVMaxPluginHooks[target]


struct SPIRVMaxPluginHooks[target: CompilationTarget](MaxPluginHooks):
    comptime warp_size = 64
    comptime barrier_fn: OptionalReg[_BarrierPluginHookFnType] = _barrier
    comptime syncwarp_fn: OptionalReg[_SyncwarpPluginHookFnType] = _syncwarp
    comptime has_warp_sum: Bool = True

    @inline(.always)
    @staticmethod
    def shuffle_idx_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        comptime assert dtype != .bool, "shuffle of bool is not supported"
        return llvm_intrinsic["llvm.spv.wave.readlane", SIMD[dtype, simd_width]](
            val, _warp_base() | (offset & 63)
        )

    @inline(.always)
    @staticmethod
    def warp_sum_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](val: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
        return llvm_intrinsic[
            "llvm.spv.wave.reduce.sum", SIMD[dtype, simd_width]
        ](val)
