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
// TargetBackend for Vulkan compute (logical SPIR-V) targets: rewrites each
// kernel into a Vulkan compute entry point (VulkanKernelABI), then emits a
// SPIR-V binary through LLVM's SPIR-V backend.
//
//===----------------------------------------------------------------------===//

#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Target/Host/HostBackend.h"
#include "Target/max_license/SPIRV/SPIRVTraits.h"
#include "Target/max_license/SPIRV/VulkanKernelABI.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <atomic>
#include <optional>
#include <string>

namespace M::KGEN {
namespace {

class SPIRVBackend final : public HostBackend {
public:
  const TargetTraits *traits() const override { return &SPIRVTraits::get(); }

  bool isCodegenInterprocedural() const override { return true; }

  bool isOffload() const override { return true; }

  std::optional<unsigned> sharedMemoryAddressSpace() const override {
    return 3; // SPIR-V Workgroup storage class
  }

  // GPU branches are far costlier than on CPU, so raise the SimplifyCFG bonus.
  llvm::SimplifyCFGOptions
  adjustSimplifyCFGOptions(llvm::SimplifyCFGOptions options) const override {
    return options.bonusInstThreshold(2);
  }

  // Vulkan shaders run neither sanitizer.
  void addSanitizers(llvm::ModulePassManager &,
                     const CompilationOptions &) const override {}

  // Recursion is not supported for GPU kernel entry functions (same as clang).
  void attachCodegenAttributes(llvm::Function *kernelEntry) const override {
    kernelEntry->addFnAttr(llvm::Attribute::NoRecurse);
  }

  // The accelerator arch (e.g. "adreno710") names a GPU, not an LLVM SPIR-V
  // processor; the SPIR-V version and Vulkan environment are in the triple.
  CompilationOptions
  adjustOptionsForTargetMachine(const CompilationOptions &options,
                                llvm::StringRef) const override {
    CompilationOptions adjusted = options;
    adjusted.targetCpu.clear();
    adjusted.targetFeatures.clear();
    return adjusted;
  }

  // A Vulkan driver consumes the SPIR-V binary directly, so `--emit asm`
  // sidecars and offload objects are both the binary module.
  ErrorOr<BufferRef> emitAssembly(llvm::Module &module,
                                  EmitContext &ctx) const override {
    return emitSPIRV(module, ctx);
  }

  ErrorOr<BufferRef> emitObject(llvm::Module &module,
                                EmitContext &ctx) const override {
    return emitSPIRV(module, ctx);
  }

  ErrorOr<BufferRef> createArchive(llvm::MutableArrayRef<BufferRef> objects,
                                   llvm::StringRef,
                                   EmitContext &) const override {
    if (objects.size() != 1)
      return Error("a Vulkan SPIR-V module holds exactly one kernel");
    return std::move(objects[0]);
  }

  // This is max only target
  bool isBaseTarget() const override { return false; }

private:
  /// Debugging aid: the path `env` names for this module's `ext` file. A
  /// directory gets one file per module, named after its kernel.
  static std::optional<std::string> debugPath(const char *env, unsigned id,
                                              llvm::StringRef kernel,
                                              llvm::StringRef ext) {
    std::optional<std::string> path = llvm::sys::Process::GetEnv(env);
    if (!path || !llvm::sys::fs::is_directory(*path))
      return path;
    std::string name = kernel.take_front(80).str();
    std::replace_if(
        name.begin(), name.end(),
        [](char c) { return !llvm::isAlnum(c) && c != '_' && c != '.'; }, '_');
    return (llvm::Twine(*path) + "/" + llvm::Twine(id) + "-" + name + ext)
        .str();
  }

  static void writeDebugFile(std::optional<std::string> path,
                             llvm::function_ref<void(llvm::raw_ostream &)> fn) {
    if (!path)
      return;
    std::error_code ec;
    llvm::raw_fd_ostream os(*path, ec);
    if (!ec)
      fn(os);
  }

  ErrorOr<BufferRef> emitSPIRV(llvm::Module &module, EmitContext &ctx) const {
    // The KGEN target's layout omits "ve" (vectors aligned like their
    // elements, as Vulkan lays them out), which KGEN cannot parse; codegen
    // gets the SPIR-V target machine's exact layout.
    module.setDataLayout(ctx.tm.createDataLayout());
    static std::atomic<unsigned> counter{0};
    unsigned id = counter++;
    std::string kernel;
    for (llvm::Function &fn : module)
      if (!fn.isDeclaration() && fn.hasFnAttribute("mojo.spirv.kernel")) {
        kernel = fn.getName().str();
        break;
      }
    // Debugging aids: the optimized kernel IR before the Vulkan rewrite, and
    // the IR that LLVM's SPIR-V codegen receives.
    writeDebugFile(debugPath("MOJO_SPIRV_DUMP_IN", id, kernel, ".in.ll"),
                   [&](llvm::raw_ostream &os) { module.print(os, nullptr); });
    if (ErrorOrSuccess err = lowerSPIRVKernelsToVulkan(module))
      return err.takeError();
    writeDebugFile(debugPath("MOJO_SPIRV_DUMP", id, kernel, ".ll"),
                   [&](llvm::raw_ostream &os) { module.print(os, nullptr); });
    // The IR is already optimized; codegen only translates it, and the Vulkan
    // driver compiles the SPIR-V again for the GPU. Optimizing codegen
    // (LLVM 24) breaks two Vulkan rules: loop strength reduction turns buffer
    // and shared-memory indices into pointer increments, which logical
    // addressing cannot express, and machine-level code motion can move an
    // instruction between an OpSelectionMerge and its branch.
    ctx.tm.setOptLevel(llvm::CodeGenOptLevel::None);
    WriteableBufferRef buf = WriteableBuffer::get();
    if (ErrorOrSuccess error =
            ctx.runLlc(module, *buf, /*createObjectFile=*/true)) {
      return Error(Twine(error.getError()) +
                   ", llc failed to codegen LLVM IR to SPIR-V");
    }
    // Debugging aid: keep every SPIR-V module, e.g. to run spirv-val on the
    // kernels of a MAX graph, which never reach the file system otherwise.
    writeDebugFile(debugPath("MOJO_SPIRV_SAVE_DIR", id, kernel, ".spv"),
                   [&](llvm::raw_ostream &os) {
                     os.write(buf->getBufferStart(), buf->getBufferSize());
                   });
    return buf;
  }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetBackend<SPIRVBackend> registerSPIRVBackend;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
