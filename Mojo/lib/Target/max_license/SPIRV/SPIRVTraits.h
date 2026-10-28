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
// Per-target metadata for Vulkan compute (logical SPIR-V) targets. A prototype
// target: kernels compile through LLVM's SPIR-V backend into Vulkan compute
// shaders, e.g. for the Qualcomm Adreno 710.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TARGET_SPIRV_SPIRVTRAITS_H
#define KGEN_TARGET_SPIRV_SPIRVTRAITS_H

#include "Target/TargetTraits.h"
#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

struct SPIRVTraits final : TargetTraits {
  llvm::StringRef name() const override { return "spirv"; }

  bool isGPU() const override { return true; }

  // Only logical SPIR-V for Vulkan (e.g. "spirv1.6-unknown-vulkan1.3-compute");
  // spirv32/spirv64 are OpenCL-style physical addressing.
  bool matches(const llvm::Triple &triple) const override {
    return triple.isSPIRVLogical() && triple.getOS() == llvm::Triple::Vulkan;
  }

  // A Vulkan driver consumes the SPIR-V binary, so the per-kernel "assembly"
  // sidecar of `--emit asm` is the binary module, ready for vkCreateShaderModule.
  llvm::StringRef getAsmExtension() const override { return ".spv"; }
  llvm::StringRef getLLVMExtension() const override { return ".spv.ll"; }
  llvm::StringRef getObjectExtension() const override { return ".spv"; }
  llvm::StringRef getBitcodeExtension() const override { return ".spv.bc"; }

  llvm::ArrayRef<EmissionKind> supportedEmissionKinds() const override {
    return commonEmissionKinds();
  }

  llvm::StringRef acceleratorSectionTitle() const override {
    return "Vulkan SPIR-V (prototype):";
  }
  llvm::ArrayRef<AcceleratorArch> supportedAcceleratorArchs() const override;

  // This is max only target
  bool isBaseTarget() const override { return false; }

  /// Shared stateless instance for the backend/lowering `traits()`.
  static const SPIRVTraits &get();
};

} // namespace M::KGEN

#endif // KGEN_TARGET_SPIRV_SPIRVTRAITS_H
