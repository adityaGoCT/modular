# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""HAL plugin hooks for the Vulkan SPIR-V target (`vulkan:adreno710`)."""

from std.collections.string.string_span import _get_kgen_string
from std.sys.info import CompilationTarget
from .._trait import HalPlugin, HalPluginHooks
from ..._machine import DeviceRef
from ...execution_config.gpu_execution_config import GPUExecutionConfiguration
from ...extensions.traits import DefaultHALExtensions


struct SpirvHalPlugin(HalPlugin):
    """`HalPlugin` for the Vulkan SPIR-V target, reached through a driver
    plugin: a plain grid and block launch, no vendor extensions.
    """

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["spirv"]()
    comptime HookFor[target: CompilationTarget] = SpirvHalPluginHooks[target]


struct SpirvHalPluginHooks[target: CompilationTarget](HalPluginHooks):
    comptime ExecutionConfigType[device: DeviceRef] = GPUExecutionConfiguration
    """A grid, block and shared-memory launch."""
    comptime extensions = DefaultHALExtensions
