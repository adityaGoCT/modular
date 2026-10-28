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
// MLIR-lowering policy for Vulkan compute (logical SPIR-V) targets. Thread
// ids, barriers and subgroup operations already arrive as `llvm.spv.*`
// intrinsics from the SPIR-V stdlib and MAX plugins, so the lowering only
// marks kernel entry points; SPIRVBackend turns each marked function into a
// Vulkan compute entry point.
//
//===----------------------------------------------------------------------===//

#include "Target/TargetLowering.h"
#include "Target/max_license/SPIRV/SPIRVTraits.h"

#include "KGENToLLVM/LLVMLoweringUtils.h"
#include "KGENToLLVM/Target/max_license/GPUCommon.h"
#include "Support/ErrorOr.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <utility>

using namespace mlir;

namespace M::KGEN {
namespace {

/// LLVM function attribute that marks a kernel entry point for SPIRVBackend.
constexpr llvm::StringLiteral kSPIRVKernelAttr = "mojo.spirv.kernel";

class SPIRVLowering final : public TargetLowering {
public:
  const TargetTraits *traits() const override { return &SPIRVTraits::get(); }

  // Aggregate kernel arguments, such as tensors holding a device pointer, are
  // passed by value rather than in memory: logical SPIR-V cannot load a
  // pointer from memory, while SPIRVBackend can give each pointer field of a
  // by-value aggregate its own storage buffer. (No override of
  // `lowerKernelArgToMemory`, unlike the other GPU targets.)

  bool needsVerification() const override { return true; }

  mlir::LogicalResult verifyOp(mlir::Operation *op) const override {
    return verifyNoGPUHeapAllocation(op);
  }

  void markExportedKernel(mlir::Operation *func) const override {
    auto funcOp = mlir::cast<mlir::LLVM::LLVMFuncOp>(func);
    mlir::Builder b(funcOp.getContext());
    llvm::SmallVector<mlir::Attribute> passthrough;
    if (mlir::ArrayAttr existing = funcOp.getPassthroughAttr())
      passthrough.append(existing.begin(), existing.end());
    passthrough.push_back(b.getArrayAttr(
        {b.getStringAttr(kSPIRVKernelAttr), b.getStringAttr("true")}));
    funcOp.setPassthroughAttr(b.getArrayAttr(passthrough));
  }

  // `MAX_THREADS_PER_BLOCK_METADATA` on this target is
  // `pop.spirv.max_work_group_size`; Vulkan fixes the workgroup size in the
  // shader, so it becomes the `mojo.spirv.numthreads` function attribute that
  // SPIRVBackend turns into the entry point's LocalSize.
  mlir::LogicalResult
  mapFuncMetadata(mlir::Operation *func, mlir::NamedAttribute attr,
                  mlir::NamedAttrList &funcAttrs,
                  llvm::SmallVectorImpl<mlir::Attribute> &passthrough,
                  bool &handled) const override {
    handled = false;
    if (attr.getName() != "pop.spirv.max_work_group_size")
      return mlir::success();
    ErrorOr<std::pair<int64_t, int64_t>> valuesOr =
        getWorkGroupSizeRange(attr.getValue());
    if (valuesOr.isError())
      return mlir::emitError(func->getLoc(), "unsupported `")
             << attr.getName() << "`: " << valuesOr.takeError();
    mlir::Builder b(func->getContext());
    passthrough.push_back(
        b.getArrayAttr({b.getStringAttr("mojo.spirv.numthreads"),
                        b.getStringAttr(std::to_string(valuesOr->second))}));
    handled = true;
    return mlir::success();
  }

  // This is max only target
  bool isBaseTarget() const override { return false; }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetLowering<SPIRVLowering> registerSPIRVLowering;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
