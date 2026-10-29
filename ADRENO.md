# Mojo on Qualcomm Adreno: a Vulkan SPIR-V target (prototype)

This branch adds an experimental GPU target, `adreno710`, to the open-source
Mojo compiler and MAX: Mojo kernels compile straight to Vulkan compute shaders
(SPIR-V) for a Qualcomm Adreno 710, through LLVM's SPIR-V backend. It is a
prototype, not a Modular product. The project around it (test kernels, the
Vulkan runners, Llama 3.2 1B on the phone, and a 26-slide presentation) is at
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
- MAX graphs, compile only (MAX's virtual-device mode, nothing runs): all 26
  test graphs give SPIR-V that passes `spirv-val` for every kernel, 211
  kernels in all: elementwise ops, cast, transpose, slice, concat, gather,
  reductions, argmax, softmax, layer_norm, four matmuls (one with B
  transposed, added with the cooperative-matrix work), conv2d, both pools, and
  a full ResNet-18 (74 kernels).
- MAX's kernels on the phone: the 43 kernels of 13 of those graphs (all
  variants of MAX's elementwise kernel for eight ops, the f32 to f16 cast,
  transpose, and the naive matmul at 256x256 f32 and f16 and a 1x2048x2048
  GEMV) run on Turnip with arguments laid out from their launch descriptions
  and match NumPy (`max_kernels/test_max_kernels.py` in mojo-adreno). The
  other graphs' kernels have not run yet.
- KHR cooperative matrices on an Adreno 840 (a Galaxy S26 Ultra in Samsung's
  Remote Test Lab, stock driver), with the second target, `adreno840`: MAX's
  cooperative-matrix matmul passes on the phone for its three test graphs, and
  standalone Mojo kernels reach 2.1 TFLOP/s in fp16 and 1.3 TFLOP/s in fp32
  at 2048x2048x2048 ([below](#cooperative-matrices-khr-for-adreno-8xx)).

## MAX on the phone: a remote device

MAX's runtime has no Vulkan backend and no Android build, so the phone joins
MAX as a remote device instead: MAX runs on the Mac (graph compiler, host code,
its HAL device layer), and a driver plugin (`M_driver_*`, in mojo-adreno's
`remote/`) forwards allocations, copies and kernel launches over `adb forward`
to a small Vulkan executor on the phone. The 25 original test graphs all run
this way and match NumPy or MAX on the Mac's CPU, ResNet-18 included (max error
5.7e-5, 8.3 s per image once its weights are on the phone; the plugin streams
requests and the phone pipelines the GPU work, so that is GPU time, mostly
MAX's naive conv2d, which runs one thread per output pixel across all output
channels):

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

## Cooperative matrices (KHR) for Adreno 8xx

A second target, `adreno840`, gives the same SPIR-V and also lets kernels use
`SPV_KHR_cooperative_matrix`: tiles that a subgroup loads, multiply-adds and
stores together, which a driver with `VK_KHR_cooperative_matrix` maps to the
GPU's matrix hardware. In Mojo they are
`max.gpu.compute.arch.mma_spirv.CoopMatrix`:

```mojo
from max.gpu.compute.arch.mma_spirv import CoopMatrix, MATRIX_A, MATRIX_B, ACCUMULATOR

var acc = CoopMatrix[DType.float16, 64, 64, ACCUMULATOR].zero()
for k in range(0, K, 16):
    var a_tile = CoopMatrix[DType.float16, 64, 16, MATRIX_A].load(a.unsafe_offset(row * K + k), K)
    var b_tile = CoopMatrix[DType.float16, 16, 64, MATRIX_B].load(b.unsafe_offset(k * N + col), N)
    acc = a_tile.multiply_add(b_tile, acc)
acc.store(c.unsafe_offset(row * N + col), N)
```

Each operation is a call to a marker function (`__mojo_spirv_coopmat_*`) that
carries the matrix as an opaque 64-bit handle. The kernel ABI pass resolves
each load's and store's pointer to a buffer or shared array and an element
index, types every handle (also through phis, such as the accumulator of the K
loop), and rewrites the markers into LLVM's `__spirv_CooperativeMatrix*KHR`
builtins and `OpFConvert`; a module that uses them switches to the Vulkan
memory model, which the extension requires. A patch to LLVM
(`llvm-spirv-cooperative-matrix-vulkan.patch`) lets shaders use those builtins,
which LLVM had recorded for OpenCL only, and makes the two extensions available
to Vulkan triples. Other kernels come out as before.

The shapes are the Adreno 840's. On a Galaxy S26 Ultra (Snapdragon 8 Elite
Gen 5, stock driver 512.842.19),
`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` lists subgroup-scope
products with M = 64 and N = 16, 32 or 64 only: fp16 x fp16 -> fp16 with
K = 16, fp32 x fp32 -> fp32 with K = 8, and 8-bit integers into 32-bit with
K = 32. There is no fp16 x fp16 -> fp32, and the driver has no
`VK_QCOM_cooperative_matrix_conversion`.

MAX's matmul takes a cooperative-matrix kernel
(`max/kernels/src/linalg/matmul/gpu/adreno/`) on `adreno840` when A, B and C
are all fp16 or all fp32, there is no epilogue, rows are contiguous, M and N
are multiples of 64 and K of 16 (fp16) or 8 (fp32): one subgroup per 64x64
tile of C, accumulating in C's type. Everything else takes the naive kernel.
Of the 840's tile widths, 64 is the fastest: in standalone kernels on the
phone (mojo-adreno's `coopmat/`), 2.1 TFLOP/s in fp16 and 1.3 TFLOP/s in fp32
at 2048x2048x2048, against 0.65 and 0.37 with 16-wide tiles.

Results. With MAX's virtual device for `vulkan:adreno840`
(`coverage.py --arch=adreno840`), all 26 test graphs compile and every kernel
passes `spirv-val` for Vulkan 1.3. The two 256x256 matmuls (fp32, fp16) and the
fp16 `a @ b.T` (which the graph compiler turns into a matmul with B transposed,
loaded column-major) take the cooperative-matrix kernel; the GEMV and
ResNet-18's classifier compile it too, and fall back to the naive kernel at
run time (one row). On the phone, mojo-adreno's `coopmat/run_coopmat.py --max`
runs those three kernels with arguments laid out from their launch
descriptions, and all three match NumPy: fp32 within 2.1e-5, fp16 with the
error of fp16 accumulation. At 256x256x256 they take 0.16-0.18 ms, 183 GFLOP/s
in fp32 and 210 in fp16, against 71 and 26 for the naive kernel on the same
phone; at that size a dispatch costs about 0.15 ms whatever it computes.
Kernels for `adreno710` come out as before (GLSL450 memory model, no
extensions).

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
| KHR cooperative matrices: Mojo API, `adreno840` target, LLVM patch, MAX matmul | `max/mojo/max/gpu/compute/arch/mma_spirv.mojo`, `max/mojo/max/_plugin/spirv/targets.mojo`, `llvm-spirv-cooperative-matrix-vulkan.patch`, `max/kernels/src/linalg/matmul/gpu/adreno/` |

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

The MAX compile test (`-- --arch=adreno840` for the cooperative-matrix
target):

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
  Mac over adb. An on-device runtime needs an Android build of MAX's runtime
  and the Vulkan plugin running locally.
- Cooperative matrices have run on one Adreno 840 (a Galaxy S26 Ultra in
  Samsung's Remote Test Lab) with one driver version; the kernel assumes that
  GPU's shapes (M = 64). fp16 products accumulate in fp16, the only fp16
  accumulator the 840 offers, so long fp16 sums lose precision. Integer
  products are not lowered yet, and each subgroup computes one tile, without
  shared-memory staging or reuse across tiles. The Adreno 710's drivers
  (Turnip, stock) have no `VK_KHR_cooperative_matrix`.
- `Mojo/lib/Elaborator/Elaborator.cpp`: a second error on a node keeps the
  first instead of aborting, so failures are reported.
