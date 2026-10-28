//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Rewrites Mojo GPU kernels into Vulkan compute entry points for LLVM's SPIR-V
// backend.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_OBJECTCOMPILER_TARGET_SPIRV_VULKANKERNELABI_H
#define KGEN_COMPILER_OBJECTCOMPILER_TARGET_SPIRV_VULKANKERNELABI_H

#include "Support/ErrorOr.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <string>
#include <vector>

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN {

/// Rewrites every function marked `mojo.spirv.kernel` in `module` into a
/// Vulkan compute entry point (`main` when the module holds one kernel).
///
/// Mojo kernels take raw pointers and scalars; a Vulkan shader takes no
/// arguments and reaches memory through descriptor-bound storage buffers with
/// logical addressing (no pointer arithmetic). So:
///  - calls are inlined, and a trap (a failed `debug_assert`) ends the
///    invocation,
///  - the arguments are flattened in order, aggregates (such as tensors,
///    passed by value, or in memory as `byval` when too large) field by field:
///    the k-th pointer becomes the storage buffer at (set 0, binding k), and
///    the j-th scalar becomes 64-bit slot j of a params storage buffer at the
///    binding after the last pointer,
///  - every address into a buffer or a shared (Workgroup) array is tracked as
///    (buffer, element index) through its GEP chain and rebuilt at each load,
///    store and atomic as `llvm.spv.resource.getpointer` or a typed GEP; a
///    pointer that a phi or select takes from several buffers also carries a
///    selector, and each access through it branches on the selector,
///  - an access of another type than the memory's elements (a struct, a
///    narrower or wider scalar) reads or writes whole elements and shifts the
///    pieces; `ptrtoint` gives each buffer its own aligned base,
///  - `llvm.spv.workgroup.size` (the block size) folds to
///    `MOJO_SPIRV_LOCAL_SIZE` ("x[,y[,z]]") when that fixes the workgroup
///    size, and is read from the params buffer otherwise (see below),
///  - multiply-high is expanded, and vectors wider than four elements, which
///    Vulkan cannot hold, are scalarized.
///
/// Each kernel gets a launch description (JSON, "abi": "mojo-vulkan-1"): its
/// arguments' sizes, and for each storage buffer and params slot the argument
/// and byte offset it comes from. Unless `MOJO_SPIRV_LOCAL_SIZE` fixes it, the
/// workgroup size is the launch's block size: specialization constants 0-2
/// set it at pipeline creation, and the kernel reads it from the params
/// buffer's last three slots.
struct VulkanModuleInfo {
  llvm::SmallVector<std::string, 1> descriptions;
  bool specializeWorkgroupSize = false;
};

ErrorOrSuccess lowerSPIRVKernelsToVulkan(llvm::Module &module,
                                         VulkanModuleInfo &info);

/// Finishes the SPIR-V that LLVM emitted for `info`'s module: embeds each
/// kernel description as an `OpModuleProcessed "mojo.vulkan.abi <json>"`, and
/// with a dynamic workgroup size adds the specialization constants (SpecId
/// 0, 1, 2) and the WorkgroupSize built-in that override LocalSize.
ErrorOr<std::vector<uint32_t>>
finishVulkanSPIRV(llvm::ArrayRef<uint32_t> words, const VulkanModuleInfo &info);

} // namespace M::KGEN

#endif // KGEN_COMPILER_OBJECTCOMPILER_TARGET_SPIRV_VULKANKERNELABI_H
