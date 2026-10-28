# Mojo on Qualcomm Adreno: a Vulkan SPIR-V target (prototype)

This branch adds an experimental GPU target, `adreno710`, to the open-source
Mojo compiler and MAX: Mojo kernels compile straight to Vulkan compute shaders
(SPIR-V) for a Qualcomm Adreno 710, through LLVM's SPIR-V backend. It is a
prototype, not a Modular product. The project around it (test kernels, the
Vulkan runners, Llama 3.2 1B on the phone, and a 24-slide presentation) is at
[adityaGoCT/mojo-adreno](https://github.com/adityaGoCT/mojo-adreno). The
presentation, with speaker notes, is also
[online](https://claude.ai/artifact/9HodHBDh22HuBYoo5UxTN6).

## Results

Measured on a moto g stylus 5G (2024), Adreno 710, Mesa Turnip (Vulkan 1.4).
A rerun after codegen started structuring the control flow with StructurizeCFG
(2026-10-28) gave the same picture: all eight kernels pass, at or below the
retargeter's times, and Llama 48 of 48 tokens at 281.0 ms a token.

- Eight test kernels (vector add, vec4, warp sum, block sum, four SGEMMs) pass
  a full output check, as fast as or faster than the earlier route that
  retargeted Mojo's AMD LLVM IR (block sum 4.9-6.3 ms vs 11-14 ms).
- Llama 3.2 1B Instruct with seven Mojo kernels: 48 of 48 tokens identical to
  PyTorch, logits within 6.2e-5, 283.6 ms a token.
- MAX graphs, compile only (MAX's virtual-device mode, nothing runs): all 25
  test graphs give SPIR-V that passes `spirv-val` for every kernel, 211
  kernels in all: elementwise ops, cast, transpose, slice, concat, gather,
  reductions, argmax, softmax, layer_norm, three matmuls, conv2d, both pools,
  and a full ResNet-18 (74 kernels).
- MAX's kernels on the phone: the 43 kernels of 13 of those graphs (all
  variants of MAX's elementwise kernel for eight ops, the f32 to f16 cast,
  transpose, and the naive matmul at 256x256 f32 and f16 and a 1x2048x2048
  GEMV) run on Turnip with arguments laid out from their launch descriptions
  and match NumPy (`max_kernels/test_max_kernels.py` in mojo-adreno). The
  other graphs' kernels have not run yet.

## MAX on the phone: a remote device

MAX's runtime has no Vulkan backend and no Android build, so the phone joins
MAX as a remote device instead: MAX runs on the Mac (graph compiler, host code,
its HAL device layer), and a driver plugin (`M_driver_*`, in mojo-adreno's
`remote/`) forwards allocations, copies and kernel launches over `adb forward`
to a small Vulkan executor on the phone. All 25 test graphs run this way and
match NumPy or MAX on the Mac's CPU, ResNet-18 included (max error 5.7e-5,
14 s, most of it one network round trip per launch):

```bash
BAZELISK_SKIP_WRAPPER=true ./bazelw build --config=disable-lint \
    --//:hal_device_context=true //max/examples/vulkan_coverage:remote //Target:CPUPlugin.so
# phone: vk_server (remote/ in mojo-adreno), then: adb forward tcp:7788 tcp:7788
MODULAR_DRIVER_PLUGINS="cpu@bazel-bin/Target/CPUPlugin.so;vulkan@/path/libM_driver_vulkan_remote.dylib" \
    ./bazelw run --config=disable-lint --//:hal_device_context=true \
    //max/examples/vulkan_coverage:remote
```

What it took in this tree: MAX's HAL device layer (`--//:hal_device_context`,
whose `--config` alias is internal) now also builds the `max` Mojo package from
its HAL overlay, so the kernel packages match the runtime's `DeviceContext`;
`vulkan` is a Mojo vendor (`DeviceAPINames.h`), and a plugin API without a
`Device::API` constant still yields its `vendor:arch`; and a `spirv` HAL plugin
gives the SPIR-V target a plain grid-and-block launch.

## How it fits together

| Piece | Files |
|---|---|
| Target identity | `Mojo/lib/Target/max_license/SPIRV/` |
| Lowering hooks: kernel marking, block-size metadata | `Mojo/lib/KGENToLLVM/Target/max_license/SPIRV/` |
| Backend: Vulkan kernel ABI pass, SPIR-V emission | `Mojo/lib/Compiler/ObjectCompiler/Target/max_license/SPIRV/` |
| stdlib plugin: SPIR-V builtins, GLSL.std.450 math | `Mojo/stdlib/std/_plugin/spirv/` |
| MAX plugin and target: 64-lane warps, barrier, shuffles, warp sum | `max/mojo/max/_plugin/spirv/` |
| New MAX hook `warp_sum_fn` | `max/mojo/max/_plugin/_trait.mojo`, `max/mojo/max/gpu/primitives/warp.mojo` |
| Matmul on GPUs without matrix units | `max/kernels/src/linalg/matmul/gpu/__init__.mojo` |
| Compile-only virtual Vulkan device | `MLRT/lib/Driver/DeviceContext/CompilationDevice.cpp` |
| MAX compile test, and graphs on the remote device (`remote.py`) | `max/examples/vulkan_coverage/` |
| MAX's HAL device layer for the target: `spirv` HAL plugin, `vulkan` vendor | `max/mojo/max/_driver/max_license/hal/_plugin/spirv/`, `MLRT/include/MLRT/Driver/DeviceContext/DeviceAPINames.h` |
| LLVM's SPIR-V backend in Mojo's LLVM build, and a structurizer fix | `bazel/public-patches/llvm.MODULE.bazel`, `llvm-spirv-structurizer-split-headers.patch` |

## Kernel ABI

A Vulkan compute shader takes no arguments, so the backend's pass
(`VulkanKernelABI.cpp`) rewrites each kernel into an entry point `main`:

- Calls are inlined, and a trap (a failed `debug_assert`) ends the
  invocation.
- The arguments are flattened in order; aggregates such as tensors are passed
  by value (or, when larger than Mojo passes in registers, in memory as
  `byval`) and split field by field. The k-th pointer becomes the storage
  buffer at (set 0, binding k); the j-th scalar becomes 64-bit slot j of a
  params storage buffer at the binding after the last pointer.
- Addresses are tracked as (buffer, element index), since logical SPIR-V has
  no pointer arithmetic. Vector loads and stores are split per element.
- A pointer that a phi or select takes from several buffers (concat's inputs)
  also carries a selector: a load reads each candidate buffer (the ones not
  selected at element 0) and selects the value; a store branches.
- An access whose type is not the buffer's element type (a struct in shared
  memory, a bool in a byte buffer) reads or writes whole elements and shifts
  the pieces. `ptrtoint` gives each buffer its own base, 2^40 apart, plus the
  byte offset, so alignment checks see aligned buffers. memset and memcpy on
  buffers become element loops.
- The workgroup size is each launch's block size, which MAX picks at the
  launch while Vulkan fixes it per pipeline: specialization constants 0, 1, 2
  (a `WorkgroupSize` built-in) set it when the runtime creates the pipeline,
  and the kernel reads `block_dim` from the last three params slots. Setting
  `MOJO_SPIRV_LOCAL_SIZE` ("x[,y[,z]]") instead fixes it in the shader (the
  kernel's `MAX_THREADS_PER_BLOCK_METADATA` if it has one), as the runners in
  mojo-adreno expect.
- Each module carries its kernel's launch description, as
  `OpModuleProcessed "mojo.vulkan.abi {json}"`: every argument's size, and for
  each storage buffer and params slot the argument and byte offset it comes
  from (and whether the kernel reads or writes the buffer), plus the
  workgroup size. A runtime lays out a launch from it; drivers ignore it.

## Build and use

```bash
./bazelw build //Mojo/tools/mojo:mojo //Mojo/stdlib/std //max/mojo/max:max
```

On a Mac with only the Command Line Tools (no Xcode.app), the repository's Bazel
wrapper refuses to run; `build/wrapper.bazelrc` can be written by hand
(`import %workspace%/build/local-resources.bazelrc`) and Bazel run with
`BAZELISK_SKIP_WRAPPER=true ./bazelw ...`.

```bash
bazel-bin/Mojo/tools/mojo/mojo build --emit asm --target-accelerator adreno710 \
    -I bazel-bin/Mojo/stdlib/std -I bazel-bin/max/mojo/max -o host.s kernel.mojo
# -> one .spv per kernel, next to host.s
```

The MAX compile test:

```bash
./bazelw run --config=disable-lint //max/examples/vulkan_coverage:coverage
```

Before codegen, wide vectors (more than four elements, as in f16 kernels)
are scalarized, and LLVM's StructurizeCFG structures the control flow: LLVM's
SPIR-V structurizer expects HLSL's nearly structured code and misplaces merge
blocks in some of Mojo's optimized kernels.

Debugging aids: `MOJO_SPIRV_DUMP_IN` and `MOJO_SPIRV_DUMP` write the LLVM IR
before and after the Vulkan rewrite, and `MOJO_SPIRV_SAVE_DIR` keeps every
SPIR-V module the backend emits; given a directory, each writes one file per
kernel.

## Limits

- Mesa Turnip (Vulkan 1.3+, 64-bit integers) only; Qualcomm's stock Vulkan 1.1
  driver and OpenCL are not covered.
- Codegen runs at -O0 to avoid an LLVM 24 SPIR-V crash (loop strength
  reduction). A second LLVM bug, which put an OpSelectionMerge before an
  unconditional branch, is patched in this branch.
- Loads
  through a pointer into several buffers also read element 0 of the others,
  which must not be empty.
- `barrier()` orders shared memory only; LLVM has no intrinsic for a
  workgroup-scope barrier on storage buffers. `warp.sum` is one subgroup
  reduction, exact with 64-lane subgroups or workgroups of at most 64.
- MAX itself does not run on the phone: it drives the phone remotely from the
  Mac, synchronously, one round trip per operation. An on-device runtime needs
  an Android build of MAX's runtime and the Vulkan plugin running locally.
- `Mojo/lib/Elaborator/Elaborator.cpp`: a second error on a node keeps the
  first instead of aborting, so failures are reported.
