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
///  - the arguments are flattened in order, aggregates (such as tensors,
///    passed by value) field by field: the k-th pointer becomes the storage
///    buffer at (set 0, binding k), and the j-th scalar becomes 64-bit slot j
///    of a params storage buffer at the binding after the last pointer,
///  - every address into a buffer or a shared (Workgroup) array is tracked as
///    (buffer, element index) through its GEP chain and rebuilt at each load,
///    store and atomic as `llvm.spv.resource.getpointer` or a typed GEP,
///  - `llvm.spv.workgroup.size` folds to the workgroup size, which Vulkan fixes
///    in the shader: `MOJO_SPIRV_LOCAL_SIZE` ("x[,y[,z]]"), 64 by default.
ErrorOrSuccess lowerSPIRVKernelsToVulkan(llvm::Module &module);

} // namespace M::KGEN

#endif // KGEN_COMPILER_OBJECTCOMPILER_TARGET_SPIRV_VULKANKERNELABI_H
