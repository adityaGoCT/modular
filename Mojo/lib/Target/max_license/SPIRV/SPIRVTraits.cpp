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

#include "Target/max_license/SPIRV/SPIRVTraits.h"

namespace M::KGEN {

const SPIRVTraits &SPIRVTraits::get() {
  static const SPIRVTraits instance;
  return instance;
}

static constexpr TargetTraits::AcceleratorArch kSPIRVAcceleratorArchs[] = {
    {"adreno710", "Qualcomm Adreno 710 (Vulkan 1.3 SPIR-V, 64-lane warps)"},
};

llvm::ArrayRef<TargetTraits::AcceleratorArch>
SPIRVTraits::supportedAcceleratorArchs() const {
  return kSPIRVAcceleratorArchs;
}

namespace {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetTraits<SPIRVTraits> registerSPIRVTraits;
#pragma GCC diagnostic pop
} // namespace

} // namespace M::KGEN
