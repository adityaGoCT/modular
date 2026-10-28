# Mojo on Qualcomm Adreno: a Vulkan SPIR-V target (prototype)

This branch adds an experimental GPU target, `adreno710`, to the open-source
Mojo compiler and MAX: Mojo kernels compile straight to Vulkan compute shaders
(SPIR-V) for a Qualcomm Adreno 710, through LLVM's SPIR-V backend. It is a
prototype, not a Modular product. The project around it (test kernels, the
Vulkan runners, Llama 3.2 1B on the phone, and a 23-slide presentation) is at
[adityaGoCT/mojo-adreno](https://github.com/adityaGoCT/mojo-adreno). The
presentation, with speaker notes, is also
[online](https://claude.ai/artifact/9HodHBDh22HuBYoo5UxTN6).

## Results

Measured on a moto g stylus 5G (2024), Adreno 710, Mesa Turnip (Vulkan 1.4):

- Eight test kernels (vector add, vec4, warp sum, block sum, four SGEMMs) pass
  a full output check, as fast as or faster than the earlier route that
  retargeted Mojo's AMD LLVM IR (block sum 4.9-6.3 ms vs 11-14 ms).
- Llama 3.2 1B Instruct with seven Mojo kernels: 48 of 48 tokens identical to
  PyTorch, logits within 6.2e-5, 283.6 ms a token.
- MAX graphs, compile only (MAX's virtual-device mode, nothing runs): 16 of 25
  test graphs give SPIR-V that passes `spirv-val` for every kernel, including
  matmul, GEMV, the elementwise ops, sum and mean.

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
| MAX compile test | `max/examples/vulkan_coverage/` |
| LLVM's SPIR-V backend in Mojo's LLVM build | `bazel/public-patches/llvm.MODULE.bazel` |

## Kernel ABI

A Vulkan compute shader takes no arguments, so the backend's pass
(`VulkanKernelABI.cpp`) rewrites each kernel into an entry point `main`:

- The arguments are flattened in order; aggregates such as tensors are passed
  by value and split field by field. The k-th pointer becomes the storage
  buffer at (set 0, binding k); the j-th scalar becomes 64-bit slot j of a
  params storage buffer at the binding after the last pointer.
- Addresses are tracked as (buffer, element index), since logical SPIR-V has
  no pointer arithmetic. Vector loads and stores are split per element.
- The workgroup size is fixed in the shader: the kernel's
  `MAX_THREADS_PER_BLOCK_METADATA`, else `MOJO_SPIRV_LOCAL_SIZE` ("x[,y[,z]]"),
  else 64.

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

Debugging aids: `MOJO_SPIRV_DUMP_IN` and `MOJO_SPIRV_DUMP` write the LLVM IR
before and after the Vulkan rewrite; `MOJO_SPIRV_SAVE_DIR` keeps every SPIR-V
module the backend emits.

## Limits

- Mesa Turnip (Vulkan 1.3+, 64-bit integers) only; Qualcomm's stock Vulkan 1.1
  driver and OpenCL are not covered.
- Codegen runs at -O0 to avoid two LLVM 24 SPIR-V bugs (loop strength reduction
  aborts; an OpSelectionMerge is misplaced), which still affects 3 MAX graphs.
- `barrier()` orders shared memory only; LLVM has no intrinsic for a
  workgroup-scope barrier on storage buffers. `warp.sum` is one subgroup
  reduction, exact with 64-lane subgroups or workgroups of at most 64.
- There is no MAX runtime for Vulkan or Android: kernels run through small C
  programs in mojo-adreno. A Vulkan driver plugin and an Android build would be
  next.
- `Mojo/lib/Elaborator/Elaborator.cpp`: a second error on a node keeps the
  first instead of aborting, so failures are reported.
