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
"""Vulkan compute (SPIR-V) GPU targets, each carrying the SPIR-V stdlib and MAX
plugins."""

from std.sys import CompilationTarget
from std._gpu.host.info import (
    AcceleratorArchitectureFamily,
    GPUInfo,
    TargetAccelerator,
    TargetAcceleratorCollection,
    TargetAcceleratorType,
    _K,
    _KB,
    _supported_archs_line,
)
from std._plugin.spirv import SPIRVPlugin

from . import SPIRVMaxPlugin

comptime SPIRV_PLUGINS = TypeList.of[SPIRVPlugin, SPIRVMaxPlugin]().values
"""The `opaque_plugin` payload of every Vulkan SPIR-V target."""


struct SpirvTargets(TargetAcceleratorCollection):
    """Vulkan SPIR-V GPU targets that the SPIR-V plugins contribute."""

    comptime vendor_name: String = "Vulkan SPIR-V"

    # fmt: off
    comptime RAW_TARGETS = TypeList.of[
        Trait=TargetAcceleratorType,
        Adreno710._with_cli_values[["adreno710"]],
    ]().values
    # fmt: on

    @staticmethod
    def normalize_target_arch(target_arch0: StaticString) -> String:
        # "vulkan:adreno710" (a MAX virtual or plugin device) -> "adreno710".
        return target_arch0.replace("vulkan:", "")

    @staticmethod
    def supported_archs_message() -> String:
        return String("  ", _supported_archs_line[Self](), "\n\n")


comptime AdrenoA7Family = AcceleratorArchitectureFamily(
    warp_size=64,
    threads_per_multiprocessor=64 * 16,
    shared_memory_per_multiprocessor=32 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""Qualcomm Adreno 7xx family, seen through Vulkan: 64-lane warps, 32 KB of
shared (Workgroup) memory per workgroup."""

comptime _adreno710_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "spirv1.6-unknown-vulkan1.3-compute", `,
        `stdlib_plugin = "spirv", `,
        `opaque_plugin = `,
        SPIRV_PLUGINS,
        `,`,
        `arch = "adreno710", `,
        `features = "", `,
        # LLVM's logical SPIR-V layout minus "ve" (vectors aligned like their
        # elements), which KGEN's parser predates; SPIRVBackend restores the
        # exact layout before codegen.
        `data_layout = "e-i64:64-n8:16:32:64-G10", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for the Adreno 710 through Vulkan 1.3 (SPIR-V 1.6),
as Mesa Turnip exposes it."""

comptime Adreno710 = TargetAccelerator[
    GPUInfo.from_family(
        family=AdrenoA7Family,
        name="Adreno 710",
        api="vulkan",
        arch_name="adreno710",
        compute=1.3,  # Vulkan version
        version="vulkan_1_3",
        sm_count=2,
    ),
    _adreno710_target,
]()
"""Qualcomm Adreno 710 GPU configuration."""
