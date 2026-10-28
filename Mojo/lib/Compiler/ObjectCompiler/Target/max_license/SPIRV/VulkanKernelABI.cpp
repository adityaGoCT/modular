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

#include "Target/max_license/SPIRV/VulkanKernelABI.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"
#include "llvm/Transforms/Scalar/StructurizeCFG.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/FixIrreducible.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LowerSwitch.h"
#include "llvm/Transforms/Utils/UnifyLoopExits.h"
#include <array>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <string>

using namespace llvm;

namespace M::KGEN {
namespace {

constexpr StringLiteral kKernelAttr = "mojo.spirv.kernel";
/// LLVM SPIR-V address spaces: Workgroup (shared memory) and StorageBuffer.
constexpr unsigned kWorkgroupAS = 3;
constexpr unsigned kStorageBufferAS = 11;
/// SPIR-V StorageClass::StorageBuffer, as a `spirv.VulkanBuffer` parameter.
constexpr unsigned kStorageBufferClass = 12;

using LocalSize = std::array<unsigned, 3>;

/// Reads the workgroup size from `MOJO_SPIRV_LOCAL_SIZE` ("x[,y[,z]]"), and
/// whether it is set.
ErrorOr<std::pair<LocalSize, bool>> getLocalSize() {
  LocalSize size = {64, 1, 1};
  std::optional<std::string> env =
      sys::Process::GetEnv("MOJO_SPIRV_LOCAL_SIZE");
  if (!env)
    return std::make_pair(size, false);
  SmallVector<StringRef> parts;
  StringRef(*env).split(parts, ',');
  if (parts.size() > 3)
    return Error("invalid MOJO_SPIRV_LOCAL_SIZE '" + Twine(*env) + "'");
  for (auto [i, part] : enumerate(parts))
    if (part.trim().getAsInteger(10, size[i]) || size[i] == 0)
      return Error("invalid MOJO_SPIRV_LOCAL_SIZE '" + Twine(*env) + "'");
  return std::make_pair(size, true);
}

std::string printed(const Type &type) {
  std::string text;
  raw_string_ostream os(text);
  type.print(os);
  return text;
}

std::string printed(const Value &value) {
  std::string text;
  raw_string_ostream os(text);
  value.print(os);
  return StringRef(text).trim().str();
}

/// Returns the index of `inst`'s pointer operand if it is a load, store or
/// atomic, else ~0u.
unsigned pointerOperandIndex(const Instruction &inst) {
  if (isa<LoadInst>(inst))
    return LoadInst::getPointerOperandIndex();
  if (isa<StoreInst>(inst))
    return StoreInst::getPointerOperandIndex();
  if (isa<AtomicRMWInst>(inst))
    return AtomicRMWInst::getPointerOperandIndex();
  if (isa<AtomicCmpXchgInst>(inst))
    return AtomicCmpXchgInst::getPointerOperandIndex();
  return ~0u;
}

/// Returns the type a load, store or atomic reads or writes.
Type *accessType(const Instruction &inst) {
  if (auto *store = dyn_cast<StoreInst>(&inst))
    return store->getValueOperand()->getType();
  if (auto *cmpxchg = dyn_cast<AtomicCmpXchgInst>(&inst))
    return cmpxchg->getCompareOperand()->getType();
  return inst.getType();
}

/// Returns the scalar type that memory reached through `root` is accessed as:
/// the element type of the loads, stores and atomics on it or on the GEPs,
/// casts, phis and selects based on it, the first scalar field of an
/// aggregate access. A byte type (from byte-offset GEPs and memcpy-like
/// accesses, or bools) loses to any other; float if the memory is never
/// accessed.
Type *inferElementType(Value *root) {
  Type *found = nullptr, *byte = nullptr;
  SmallVector<Value *> work{root};
  SmallPtrSet<Value *, 16> seen;
  while (!work.empty()) {
    Value *value = work.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    for (User *user : value->users()) {
      Type *type = nullptr;
      if (auto *load = dyn_cast<LoadInst>(user))
        type = load->getType();
      else if (auto *store = dyn_cast<StoreInst>(user);
               store && store->getPointerOperand() == value)
        type = store->getValueOperand()->getType();
      else if (auto *rmw = dyn_cast<AtomicRMWInst>(user))
        type = rmw->getType();
      else if (auto *cmpxchg = dyn_cast<AtomicCmpXchgInst>(user))
        type = cmpxchg->getCompareOperand()->getType();
      else if (isa<GetElementPtrInst, AddrSpaceCastInst, BitCastInst, PHINode,
                   SelectInst>(user))
        work.push_back(user);
      if (!type)
        continue;
      for (;;) {
        if (auto *aggregate = dyn_cast<StructType>(type);
            aggregate && aggregate->getNumElements())
          type = aggregate->getElementType(0);
        else if (auto *array = dyn_cast<ArrayType>(type))
          type = array->getElementType();
        else
          break;
      }
      type = type->getScalarType();
      if (type->isAggregateType() || type->isPointerTy())
        continue;
      if (type->isIntegerTy(1))
        type = Type::getInt8Ty(type->getContext());
      if (type->isIntegerTy(8))
        byte = type;
      else if (!found)
        found = type;
    }
  }
  if (found)
    return found;
  return byte ? byte : Type::getFloatTy(root->getContext());
}

class KernelRewriter {
public:
  KernelRewriter(Module &module, Function &kernel, StringRef entryName,
                 LocalSize localSize, bool dynamicSize)
      : module(module), kernel(kernel), entryName(entryName),
        localSize(localSize), dynamicSize(dynamicSize),
        dl(module.getDataLayout()), ctx(module.getContext()),
        i32(Type::getInt32Ty(ctx)) {}

  ErrorOrSuccess run();
  /// The kernel's launch description (JSON), once `run` succeeded.
  const std::string &getDescription() const { return description; }

private:
  /// Memory a pointer reaches: a storage buffer or a shared array, or a group
  /// of them (`choices`, two or more such roots with one element type) for a
  /// pointer that a phi or select takes from several.
  struct Root {
    Value *handle = nullptr;           // the buffer's resource handle, or
    GlobalVariable *shared = nullptr;  // the Workgroup array
    Type *elementType = nullptr;
    Type *storageType = nullptr;       // handle or array type
    SmallVector<unsigned, 2> choices;  // a group's roots, sorted
    bool read = false, written = false;
  };
  /// A pointer as (root, i32 element index); a null index means element 0.
  /// Into a group, `selector` (i32) is the position of the actual root among
  /// the group's choices.
  struct Address {
    unsigned root;
    Value *index;
    Value *selector = nullptr;
  };
  /// One flattened piece of an argument: the argument itself, or a pointer or
  /// scalar field of an aggregate one, with the values that read it.
  struct Leaf {
    Type *type;
    SmallVector<Value *> values;
    unsigned arg = 0;     // the argument it belongs to
    uint64_t offset = 0;  // its byte offset in the argument's value
  };

  ErrorOrSuccess flattenArgument(Argument &arg, SmallVectorImpl<Leaf> &leaves);
  ErrorOrSuccess flattenInMemory(Argument &arg, Type *type,
                                 SmallVectorImpl<Leaf> &leaves);

  Value *createHandle(IRBuilder<> &b, Type *handleType, unsigned binding,
                      StringRef name);
  Value *getPointer(IRBuilder<> &b, const Root &root, Value *index);
  ErrorOrSuccess addSharedRoots(Function &entry, IRBuilder<> &b);
  ErrorOr<Value *> elementIndex(IRBuilder<> &b, GetElementPtrInst &gep,
                                Type *elementType);
  Value *scaledIndex(IRBuilder<> &b, Value *value, int64_t bytes,
                     int64_t size);
  void rootsOf(Value *value, SmallPtrSetImpl<Value *> &visiting,
               SmallSetVector<unsigned, 4> &found);
  void addChoices(const Address &address, SmallSetVector<unsigned, 4> &found);
  ErrorOr<unsigned> groupFor(ArrayRef<unsigned> choices, Instruction &inst);
  Value *selectorIn(IRBuilder<> &b, const Address &address, unsigned group);
  Value *addressAsInt(IRBuilder<> &b, const Address &address);
  ErrorOrSuccess visit(Instruction &inst);
  ErrorOrSuccess rewriteAccess(Instruction &inst, unsigned root, Value *index);
  void retypeAccess(IRBuilder<> &b, Instruction &inst, Type *elementType);
  bool splitVectorAccess(IRBuilder<> &b, Instruction &inst, const Root &root,
                         Value *index);
  bool needsAccessByParts(Type *type, Type *elementType);
  ErrorOrSuccess accessByParts(IRBuilder<> &b, Instruction &inst,
                               const Root &root, Value *index);
  ErrorOrSuccess finishPhis();
  ErrorOrSuccess expandGroupAccesses();
  ErrorOrSuccess expandMemIntrinsics();
  ErrorOrSuccess eraseDead();

  Module &module;
  Function &kernel;
  StringRef entryName;
  LocalSize localSize;
  /// Whether the workgroup size is the launch's block size, set at pipeline
  /// creation (specialization constants), rather than fixed in the shader.
  bool dynamicSize;
  /// With a dynamic size, the launch's block size, read from the params.
  std::array<Value *, 3> blockDims{};
  std::string description;
  const DataLayout &dl;
  LLVMContext &ctx;
  Type *i32;
  SmallVector<Root> roots;
  DenseMap<Value *, Address> addresses;
  /// Pointer phis and the index (and, into a group, selector) phis that
  /// replace them, filled in once every incoming address (some along back
  /// edges) is known.
  SmallVector<std::tuple<PHINode *, PHINode *, PHINode *>> phis;
  /// Accesses through a group address, and memset/memcpy on buffers or
  /// shared arrays, expanded once the traversal is done.
  SmallVector<std::pair<Instruction *, Address>> groupAccesses;
  SmallVector<MemIntrinsic *> memIntrinsics;
  SmallVector<Instruction *> dead;
  SmallVector<GlobalVariable *> retypedGlobals;
};

/// Appends `arg`'s leaves to `leaves`: the argument itself, or each field of
/// an aggregate (struct or array, recursively) in field order. An aggregate is
/// only read field by field, through `extractvalue`s, which become the leaves'
/// values.
ErrorOrSuccess KernelRewriter::flattenArgument(Argument &arg,
                                               SmallVectorImpl<Leaf> &leaves) {
  if (arg.hasByValAttr())
    return flattenInMemory(arg, arg.getParamByValType(), leaves);
  Type *type = arg.getType();
  if (!type->isAggregateType()) {
    leaves.push_back({type, {&arg}, arg.getArgNo(), 0});
    return {};
  }
  struct Field {
    SmallVector<unsigned> path;
    Type *type;
    uint64_t offset;
  };
  SmallVector<Field> fields;
  SmallVector<unsigned> path;
  std::function<void(Type *, uint64_t)> collect = [&](Type *field,
                                                      uint64_t offset) {
    if (auto *structType = dyn_cast<StructType>(field)) {
      const StructLayout *layout = dl.getStructLayout(structType);
      for (unsigned i = 0, e = structType->getNumElements(); i != e; ++i) {
        path.push_back(i);
        collect(structType->getElementType(i),
                offset + layout->getElementOffset(i));
        path.pop_back();
      }
    } else if (auto *arrayType = dyn_cast<ArrayType>(field)) {
      uint64_t stride = dl.getTypeAllocSize(arrayType->getElementType());
      for (uint64_t i = 0, e = arrayType->getNumElements(); i != e; ++i) {
        path.push_back(static_cast<unsigned>(i));
        collect(arrayType->getElementType(), offset + i * stride);
        path.pop_back();
      }
    } else {
      fields.push_back({path, field, offset});
    }
  };
  collect(type, 0);
  size_t first = leaves.size();
  for (auto &field : fields)
    leaves.push_back({field.type, {}, arg.getArgNo(), field.offset});

  SmallVector<std::pair<Value *, SmallVector<unsigned>>> work{{&arg, {}}};
  while (!work.empty()) {
    auto [value, prefix] = work.pop_back_val();
    for (User *user : value->users()) {
      auto *extract = dyn_cast<ExtractValueInst>(user);
      if (!extract)
        return Error("unsupported use of an aggregate kernel argument (only "
                     "its fields can be read): " +
                     printed(*user));
      SmallVector<unsigned> fieldPath(prefix);
      fieldPath.append(extract->idx_begin(), extract->idx_end());
      dead.push_back(extract);
      if (extract->getType()->isAggregateType()) {
        work.push_back({extract, fieldPath});
        continue;
      }
      auto it = find_if(fields, [&](auto &field) {
        return ArrayRef<unsigned>(field.path) == ArrayRef<unsigned>(fieldPath);
      });
      if (it == fields.end())
        return Error("unknown field of an aggregate kernel argument: " +
                     printed(*extract));
      leaves[first + (it - fields.begin())].values.push_back(extract);
    }
  }
  return {};
}

/// Appends the leaves of `arg`, a `byval` pointer to an aggregate of `type`
/// (too large to pass in registers, so the kernel receives it in memory), in
/// field order exactly as if it were passed by value. The kernel only reads
/// it, by loads at constant offsets, which become the leaves' values.
ErrorOrSuccess KernelRewriter::flattenInMemory(Argument &arg, Type *type,
                                               SmallVectorImpl<Leaf> &leaves) {
  SmallVector<std::pair<uint64_t, Type *>> fields;
  std::function<void(Type *, uint64_t)> collect = [&](Type *field,
                                                      uint64_t offset) {
    if (auto *structType = dyn_cast<StructType>(field)) {
      const StructLayout *layout = dl.getStructLayout(structType);
      for (unsigned i = 0, e = structType->getNumElements(); i != e; ++i)
        collect(structType->getElementType(i),
                offset + layout->getElementOffset(i));
    } else if (auto *arrayType = dyn_cast<ArrayType>(field)) {
      uint64_t stride = dl.getTypeAllocSize(arrayType->getElementType());
      for (uint64_t i = 0, e = arrayType->getNumElements(); i != e; ++i)
        collect(arrayType->getElementType(), offset + i * stride);
    } else {
      fields.push_back({offset, field});
    }
  };
  collect(type, 0);
  size_t first = leaves.size();
  for (auto &field : fields)
    leaves.push_back({field.second, {}, arg.getArgNo(), field.first});

  SmallVector<std::pair<Value *, int64_t>> work{{&arg, 0}};
  while (!work.empty()) {
    auto [value, offset] = work.pop_back_val();
    for (User *user : value->users()) {
      if (auto *gep = dyn_cast<GetElementPtrInst>(user)) {
        APInt constant(dl.getIndexTypeSizeInBits(gep->getType()), 0);
        if (!gep->accumulateConstantOffset(dl, constant))
          return Error("unsupported variable index into a kernel argument "
                       "passed in memory: " +
                       printed(*gep));
        dead.push_back(gep);
        work.push_back({gep, offset + constant.getSExtValue()});
        continue;
      }
      auto *load = dyn_cast<LoadInst>(user);
      if (!load || !load->isSimple() || load->getType()->isAggregateType())
        return Error("unsupported use of a kernel argument passed in memory "
                     "(only its fields can be read): " +
                     printed(*user));
      auto it = find_if(fields, [&](auto &field) {
        return static_cast<int64_t>(field.first) == offset &&
               field.second->isPointerTy() == load->getType()->isPointerTy() &&
               dl.getTypeStoreSize(field.second) ==
                   dl.getTypeStoreSize(load->getType());
      });
      if (it == fields.end())
        return Error("load of no field of a kernel argument passed in "
                     "memory: " +
                     printed(*load));
      leaves[first + (it - fields.begin())].values.push_back(load);
      dead.push_back(load);
    }
  }
  return {};
}

Value *KernelRewriter::createHandle(IRBuilder<> &b, Type *handleType,
                                    unsigned binding, StringRef name) {
  // LLVM's SPIR-V backend names each resource from a string (null crashes it).
  Constant *nameData = ConstantDataArray::getString(ctx, name);
  auto *nameVar = new GlobalVariable(
      module, nameData->getType(), /*isConstant=*/true,
      GlobalValue::PrivateLinkage, nameData, "." + name, nullptr,
      GlobalValue::NotThreadLocal, /*AddressSpace=*/0);
  nameVar->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  Function *fn = Intrinsic::getOrInsertDeclaration(
      &module, Intrinsic::spv_resource_handlefrombinding, {handleType});
  // (descriptor set, binding, array size, array index, name)
  return b.CreateCall(fn, {b.getInt32(0), b.getInt32(binding), b.getInt32(1),
                           b.getInt32(0), nameVar});
}

Value *KernelRewriter::getPointer(IRBuilder<> &b, const Root &root,
                                  Value *index) {
  if (!index)
    index = b.getInt32(0);
  if (root.shared)
    return b.CreateInBoundsGEP(root.storageType, root.shared,
                               {b.getInt32(0), index});
  Function *fn = Intrinsic::getOrInsertDeclaration(
      &module, Intrinsic::spv_resource_getpointer,
      {PointerType::get(ctx, kStorageBufferAS), root.storageType, i32});
  return b.CreateCall(fn, {root.handle, index});
}

ErrorOrSuccess KernelRewriter::addSharedRoots(Function &entry,
                                              IRBuilder<> &b) {
  SmallVector<GlobalVariable *> shared;
  for (GlobalVariable &global : module.globals())
    if (global.getAddressSpace() == kWorkgroupAS && !global.use_empty())
      shared.push_back(&global);
  for (GlobalVariable *global : shared) {
    // Constant-expression GEPs on the array become instructions, so every
    // address into it is visited below.
    convertUsersOfConstantsToInstructions({global}, &entry);
    Type *valueType = global->getValueType();
    Type *elementType = valueType;
    uint64_t count = 1;
    if (auto *array = dyn_cast<ArrayType>(valueType)) {
      elementType = array->getElementType();
      count = array->getNumElements();
    }
    Type *accessed = inferElementType(global);
    if (elementType->isAggregateType() || elementType != accessed) {
      // A byte array (or a nested aggregate) accessed as another type is
      // retyped to an array of that type over the same bytes.
      uint64_t bytes = dl.getTypeAllocSize(valueType);
      uint64_t size = dl.getTypeAllocSize(accessed);
      if (accessed->isAggregateType() || bytes % size)
        return Error("unsupported shared memory layout: " + printed(*global));
      elementType = accessed;
      count = bytes / size;
      auto *retyped = new GlobalVariable(
          module, ArrayType::get(elementType, count), /*isConstant=*/false,
          GlobalValue::InternalLinkage, PoisonValue::get(ArrayType::get(
                                            elementType, count)),
          global->getName() + ".vk", nullptr, GlobalValue::NotThreadLocal,
          kWorkgroupAS);
      retyped->setAlignment(global->getAlign());
      global->replaceAllUsesWith(retyped);
      retypedGlobals.push_back(global);
      global = retyped;
    }
    roots.push_back(
        {nullptr, global, elementType, ArrayType::get(elementType, count), {}});
    addresses[global] = {static_cast<unsigned>(roots.size() - 1), nullptr};
  }
  return {};
}

/// Returns `value * bytes / size` as an i32 element count. When `bytes` is not
/// a multiple of the element size, the multiple usually hides in `value`
/// itself (byte-offset GEPs index by `k << 6` or `k * 64`), so shifts,
/// constant multiplies, adds and extensions are looked through; anything else
/// takes the exact 64-bit division.
Value *KernelRewriter::scaledIndex(IRBuilder<> &b, Value *value,
                                   int64_t bytes, int64_t size) {
  if (bytes % size == 0) {
    Value *term = b.CreateSExtOrTrunc(value, i32);
    if (bytes == size)
      return term;
    return b.CreateMul(term, b.getInt32(static_cast<uint32_t>(bytes / size)));
  }
  if (auto *constant = dyn_cast<ConstantInt>(value))
    return b.getInt32(
        static_cast<uint32_t>(constant->getSExtValue() * bytes / size));
  if (isa<SExtInst, ZExtInst>(value))
    return scaledIndex(b, cast<Instruction>(value)->getOperand(0), bytes, size);
  if (auto *op = dyn_cast<BinaryOperator>(value)) {
    auto *rhs = dyn_cast<ConstantInt>(op->getOperand(1));
    switch (op->getOpcode()) {
    case Instruction::Shl:
      if (rhs && rhs->getZExtValue() < 32)
        return scaledIndex(b, op->getOperand(0), bytes << rhs->getZExtValue(),
                           size);
      break;
    case Instruction::Mul:
      if (rhs)
        return scaledIndex(b, op->getOperand(0), bytes * rhs->getSExtValue(),
                           size);
      break;
    case Instruction::Add:
      return b.CreateAdd(scaledIndex(b, op->getOperand(0), bytes, size),
                         scaledIndex(b, op->getOperand(1), bytes, size));
    case Instruction::Or:
      if (cast<PossiblyDisjointInst>(op)->isDisjoint())
        return b.CreateAdd(scaledIndex(b, op->getOperand(0), bytes, size),
                           scaledIndex(b, op->getOperand(1), bytes, size));
      break;
    default:
      break;
    }
  }
  Value *offset = b.CreateMul(
      value, ConstantInt::get(value->getType(), static_cast<uint64_t>(bytes)));
  offset = isPowerOf2_64(static_cast<uint64_t>(size))
               ? b.CreateAShr(offset, Log2_64(static_cast<uint64_t>(size)), "",
                              /*isExact=*/true)
               : b.CreateExactSDiv(
                     offset, ConstantInt::get(offset->getType(),
                                              static_cast<uint64_t>(size)));
  return b.CreateSExtOrTrunc(offset, i32);
}

/// Returns the i32 element index `gep` adds to its base: each variable index
/// times its stride in elements, plus the constant offset in elements. Working
/// from the GEP's terms, truncated to 32 bits first, avoids computing a 64-bit
/// byte offset only to divide it again; Adreno has no 64-bit ALU, and codegen
/// (at -O0) would not fold the round trip.
ErrorOr<Value *> KernelRewriter::elementIndex(IRBuilder<> &b,
                                              GetElementPtrInst &gep,
                                              Type *elementType) {
  unsigned bits = dl.getIndexTypeSizeInBits(gep.getType());
  SmallMapVector<Value *, APInt, 4> variable;
  APInt constant(bits, 0);
  if (!cast<GEPOperator>(gep).collectOffset(dl, bits, variable, constant))
    return Error("unsupported address computation: " + printed(gep));
  int64_t size = static_cast<int64_t>(dl.getTypeAllocSize(elementType));
  Value *index = nullptr;
  auto add = [&](Value *term) {
    index = index ? b.CreateAdd(index, term) : term;
  };
  for (auto &[value, stride] : variable)
    add(scaledIndex(b, value, stride.getSExtValue(), size));
  int64_t offset = constant.getSExtValue();
  if (offset % size)
    return Error("address is not element-aligned: " + printed(gep));
  if (offset || !index)
    add(b.getInt32(static_cast<uint32_t>(offset / size)));
  return index;
}

/// Adds the buffers or shared arrays that `address` may point into to `found`.
void KernelRewriter::addChoices(const Address &address,
                                SmallSetVector<unsigned, 4> &found) {
  const Root &root = roots[address.root];
  if (root.choices.empty())
    found.insert(address.root);
  else
    found.insert(root.choices.begin(), root.choices.end());
}

/// Collects into `found` the buffers or shared arrays that `value`, a
/// pointer, may address, following GEPs, casts, phis and selects, before they
/// are visited.
void KernelRewriter::rootsOf(Value *value, SmallPtrSetImpl<Value *> &visiting,
                             SmallSetVector<unsigned, 4> &found) {
  if (auto it = addresses.find(value); it != addresses.end())
    return addChoices(it->second, found);
  if (!value->getType()->isPointerTy() || !visiting.insert(value).second)
    return; // not an address, or a cycle back to a phi being resolved
  if (auto *gep = dyn_cast<GEPOperator>(value))
    return rootsOf(gep->getPointerOperand(), visiting, found);
  if (isa<AddrSpaceCastInst, BitCastInst>(value))
    return rootsOf(cast<Instruction>(value)->getOperand(0), visiting, found);
  if (auto *phi = dyn_cast<PHINode>(value)) {
    for (Value *input : phi->incoming_values())
      rootsOf(input, visiting, found);
  } else if (auto *select = dyn_cast<SelectInst>(value)) {
    rootsOf(select->getTrueValue(), visiting, found);
    rootsOf(select->getFalseValue(), visiting, found);
  }
}

/// Returns the group root of `choices`, two or more buffers or shared arrays,
/// for a pointer that `inst` (a phi or select) takes from any of them. Their
/// element types must agree: an access through the group is one access per
/// choice, with one element index.
ErrorOr<unsigned> KernelRewriter::groupFor(ArrayRef<unsigned> choices,
                                           Instruction &inst) {
  SmallVector<unsigned, 2> sorted(choices);
  sort(sorted);
  for (auto [i, root] : enumerate(roots))
    if (ArrayRef<unsigned>(root.choices) == ArrayRef<unsigned>(sorted))
      return static_cast<unsigned>(i);
  Type *elementType = roots[sorted.front()].elementType;
  for (unsigned choice : sorted)
    if (roots[choice].elementType != elementType)
      return Error("pointer into buffers or shared arrays of different "
                   "element types: " +
                   printed(inst));
  Root group;
  group.elementType = elementType;
  group.choices = std::move(sorted);
  roots.push_back(std::move(group));
  return static_cast<unsigned>(roots.size() - 1);
}

/// Returns the position of `address`'s actual root among the choices of
/// `group`, as an i32: the selector of the same pointer as an address into
/// `group`.
Value *KernelRewriter::selectorIn(IRBuilder<> &b, const Address &address,
                                  unsigned group) {
  if (address.root == group)
    return address.selector;
  ArrayRef<unsigned> choices = roots[group].choices;
  auto position = [&](unsigned root) {
    return b.getInt32(
        static_cast<uint32_t>(find(choices, root) - choices.begin()));
  };
  ArrayRef<unsigned> from = roots[address.root].choices;
  if (from.empty())
    return position(address.root);
  // An address into a smaller group: map each of its positions.
  Value *selector = position(from.back());
  for (size_t i = from.size() - 1; i-- > 0;)
    selector = b.CreateSelect(
        b.CreateICmpEQ(address.selector,
                       b.getInt32(static_cast<uint32_t>(i))),
        position(from[i]), selector);
  return selector;
}

/// Returns the integer value of a pointer into a buffer or shared array, for
/// `ptrtoint` (alignment checks, pointer differences) and pointer compares.
/// Logical addressing has no addresses, so each root gets its own base, 2^40
/// apart (every base looks aligned), plus the byte offset of the element.
Value *KernelRewriter::addressAsInt(IRBuilder<> &b, const Address &address) {
  const Root &root = roots[address.root];
  Type *i64 = b.getInt64Ty();
  auto base = [&](unsigned r) {
    return ConstantInt::get(i64, (static_cast<uint64_t>(r) + 1) << 40);
  };
  Value *value;
  if (root.choices.empty()) {
    value = base(address.root);
  } else {
    value = base(root.choices.back());
    for (size_t i = root.choices.size() - 1; i-- > 0;)
      value = b.CreateSelect(
          b.CreateICmpEQ(address.selector,
                         b.getInt32(static_cast<uint32_t>(i))),
          base(root.choices[i]), value);
  }
  if (address.index)
    value = b.CreateAdd(
        value, b.CreateMul(b.CreateSExt(address.index, i64),
                           ConstantInt::get(i64, dl.getTypeAllocSize(
                                                     root.elementType))));
  return value;
}

ErrorOrSuccess KernelRewriter::visit(Instruction &inst) {
  IRBuilder<> b(&inst);
  auto addressOf = [&](Value *value) -> std::optional<Address> {
    auto it = addresses.find(value);
    if (it == addresses.end())
      return std::nullopt;
    return it->second;
  };

  // A phi or select of addresses into one buffer or array becomes a phi or
  // select of element indices; into several, also of selectors.
  if (auto *phi = dyn_cast<PHINode>(&inst); phi && phi->getType()->isPointerTy()) {
    SmallPtrSet<Value *, 8> visiting;
    SmallSetVector<unsigned, 4> found;
    rootsOf(phi, visiting, found);
    if (found.empty())
      return {};
    unsigned root = found.front();
    PHINode *selector = nullptr;
    if (found.size() > 1) {
      ErrorOr<unsigned> group = groupFor(found.getArrayRef(), inst);
      if (group.isError())
        return group.takeError();
      root = *group;
      selector = PHINode::Create(i32, phi->getNumIncomingValues(), "",
                                 phi->getIterator());
    }
    PHINode *index =
        PHINode::Create(i32, phi->getNumIncomingValues(), "", phi->getIterator());
    addresses[phi] = {root, index, selector};
    phis.push_back({phi, index, selector});
    dead.push_back(phi);
    return {};
  }
  if (auto *select = dyn_cast<SelectInst>(&inst);
      select && select->getType()->isPointerTy()) {
    std::optional<Address> t = addressOf(select->getTrueValue());
    std::optional<Address> f = addressOf(select->getFalseValue());
    if (!t && !f)
      return {};
    if (!t || !f)
      return Error("select between a buffer or shared-memory pointer and "
                   "another pointer (Vulkan has logical addressing): " +
                   printed(inst));
    unsigned root = t->root;
    if (t->root != f->root) {
      SmallSetVector<unsigned, 4> found;
      addChoices(*t, found);
      addChoices(*f, found);
      ErrorOr<unsigned> group = groupFor(found.getArrayRef(), inst);
      if (group.isError())
        return group.takeError();
      root = *group;
    }
    Value *zero = b.getInt32(0);
    Value *selector = nullptr;
    if (!roots[root].choices.empty())
      selector = b.CreateSelect(select->getCondition(),
                                selectorIn(b, *t, root),
                                selectorIn(b, *f, root));
    addresses[select] = {root,
                         b.CreateSelect(select->getCondition(),
                                        t->index ? t->index : zero,
                                        f->index ? f->index : zero),
                         selector};
    dead.push_back(select);
    return {};
  }

  if (auto *gep = dyn_cast<GetElementPtrInst>(&inst)) {
    std::optional<Address> base = addressOf(gep->getPointerOperand());
    if (!base)
      return {};
    ErrorOr<Value *> index =
        elementIndex(b, *gep, roots[base->root].elementType);
    if (index.isError())
      return index.takeError();
    Value *combined = *index;
    if (base->index)
      combined = b.CreateAdd(base->index, combined);
    addresses[gep] = {base->root, combined, base->selector};
    dead.push_back(gep);
    return {};
  }
  if (isa<PtrToIntInst, PtrToAddrInst>(inst)) {
    if (std::optional<Address> address = addressOf(inst.getOperand(0))) {
      inst.replaceAllUsesWith(
          b.CreateZExtOrTrunc(addressAsInt(b, *address), inst.getType()));
      dead.push_back(&inst);
    }
    return {};
  }
  if (auto *cmp = dyn_cast<ICmpInst>(&inst);
      cmp && cmp->getOperand(0)->getType()->isPointerTy()) {
    std::optional<Address> lhs = addressOf(cmp->getOperand(0));
    std::optional<Address> rhs = addressOf(cmp->getOperand(1));
    if (!lhs && !rhs)
      return {};
    auto asInt = [&](Value *operand,
                     const std::optional<Address> &address) -> Value * {
      if (address)
        return addressAsInt(b, *address);
      if (isa<ConstantPointerNull>(operand))
        return b.getInt64(0);
      return nullptr;
    };
    Value *l = asInt(cmp->getOperand(0), lhs);
    Value *r = asInt(cmp->getOperand(1), rhs);
    if (!l || !r)
      return Error("compare of a buffer or shared-memory pointer with "
                   "another pointer (Vulkan has logical addressing): " +
                   printed(inst));
    cmp->replaceAllUsesWith(b.CreateICmp(cmp->getPredicate(), l, r));
    dead.push_back(cmp);
    return {};
  }
  if (isa<AddrSpaceCastInst, BitCastInst>(inst)) {
    if (std::optional<Address> base = addressOf(inst.getOperand(0))) {
      addresses[&inst] = *base;
      dead.push_back(&inst);
    }
    return {};
  }
  if (auto *intrinsic = dyn_cast<IntrinsicInst>(&inst)) {
    if (intrinsic->getIntrinsicID() == Intrinsic::spv_workgroup_size) {
      auto *dim = dyn_cast<ConstantInt>(intrinsic->getArgOperand(0));
      if (!dim || dim->getZExtValue() > 2)
        return Error("non-constant workgroup size dimension: " +
                     printed(inst));
      intrinsic->replaceAllUsesWith(
          dynamicSize ? b.CreateZExtOrTrunc(blockDims[dim->getZExtValue()],
                                            intrinsic->getType())
                      : ConstantInt::get(intrinsic->getType(),
                                         localSize[dim->getZExtValue()]));
      dead.push_back(intrinsic);
      return {};
    }
    // Lifetime markers and alignment assumptions on a rewritten pointer are
    // hints that logical addressing cannot express; drop them.
    if ((intrinsic->isLifetimeStartOrEnd() ||
         intrinsic->getIntrinsicID() == Intrinsic::assume) &&
        any_of(intrinsic->operands(),
               [&](Value *operand) { return addressOf(operand).has_value(); })) {
      dead.push_back(intrinsic);
      return {};
    }
    // <0, 1, 2, ...>: LLVM's SPIR-V backend cannot select it.
    if (intrinsic->getIntrinsicID() == Intrinsic::stepvector) {
      auto *type = dyn_cast<FixedVectorType>(intrinsic->getType());
      if (!type)
        return Error("unsupported scalable vector: " + printed(inst));
      SmallVector<Constant *> steps;
      for (unsigned i = 0, e = type->getNumElements(); i != e; ++i)
        steps.push_back(ConstantInt::get(type->getElementType(), i));
      intrinsic->replaceAllUsesWith(ConstantVector::get(steps));
      dead.push_back(intrinsic);
      return {};
    }
    if (auto *memory = dyn_cast<MemIntrinsic>(intrinsic);
        memory && any_of(memory->args(), [&](Value *operand) {
          return addressOf(operand).has_value();
        })) {
      memIntrinsics.push_back(memory);
      return {};
    }
    // The high half of a product (Mojo's division by a constant): LLVM's
    // SPIR-V backend selects it for OpenCL only. Up to 32 bits it is a
    // double-width product; 64 bits (unsigned) is built from 32-bit halves.
    Intrinsic::ID id = intrinsic->getIntrinsicID();
    if (id == Intrinsic::umulh || id == Intrinsic::smulh) {
      Type *type = intrinsic->getType();
      unsigned bits = type->getScalarSizeInBits();
      Value *x = intrinsic->getArgOperand(0), *y = intrinsic->getArgOperand(1);
      Value *high;
      if (bits <= 32) {
        Type *wide = type->getWithNewBitWidth(2 * bits);
        auto extend = [&](Value *value) {
          return id == Intrinsic::smulh ? b.CreateSExt(value, wide)
                                        : b.CreateZExt(value, wide);
        };
        high = b.CreateTrunc(
            b.CreateLShr(b.CreateMul(extend(x), extend(y)), bits), type);
      } else if (bits == 64 && id == Intrinsic::umulh) {
        Constant *low32 = ConstantInt::get(type, 0xffffffffu);
        auto lo = [&](Value *value) { return b.CreateAnd(value, low32); };
        auto hi = [&](Value *value) { return b.CreateLShr(value, 32); };
        Value *ll = b.CreateMul(lo(x), lo(y));
        Value *lh = b.CreateMul(lo(x), hi(y));
        Value *hl = b.CreateMul(hi(x), lo(y));
        Value *hh = b.CreateMul(hi(x), hi(y));
        Value *mid = b.CreateAdd(b.CreateAdd(hi(ll), lo(lh)), lo(hl));
        high = b.CreateAdd(b.CreateAdd(hh, hi(lh)),
                           b.CreateAdd(hi(hl), hi(mid)));
      } else {
        return Error("unsupported multiply-high: " + printed(inst));
      }
      intrinsic->replaceAllUsesWith(high);
      dead.push_back(intrinsic);
      return {};
    }
  }

  // Vulkan limits atomic and fence memory scopes to device, workgroup,
  // subgroup and invocation, and forbids sequentially consistent ordering.
  // LLVM's default (system) scope would become CrossDevice, so it narrows to
  // device, the widest Vulkan allows; seq_cst weakens to acquire-release.
  auto narrowScope = [&](auto *op) {
    if (op->getSyncScopeID() == SyncScope::System)
      op->setSyncScopeID(ctx.getOrInsertSyncScopeID("device"));
  };
  auto weaken = [](AtomicOrdering ordering) {
    return ordering == AtomicOrdering::SequentiallyConsistent
               ? AtomicOrdering::AcquireRelease
               : ordering;
  };
  if (auto *rmw = dyn_cast<AtomicRMWInst>(&inst)) {
    narrowScope(rmw);
    rmw->setOrdering(weaken(rmw->getOrdering()));
  } else if (auto *cmpxchg = dyn_cast<AtomicCmpXchgInst>(&inst)) {
    narrowScope(cmpxchg);
    cmpxchg->setSuccessOrdering(weaken(cmpxchg->getSuccessOrdering()));
    if (cmpxchg->getFailureOrdering() == AtomicOrdering::SequentiallyConsistent)
      cmpxchg->setFailureOrdering(AtomicOrdering::Acquire);
  } else if (auto *fence = dyn_cast<FenceInst>(&inst)) {
    narrowScope(fence);
    fence->setOrdering(weaken(fence->getOrdering()));
  }

  // Memory accesses: rebuild the pointer operand from its (root, index).
  unsigned pointerOperand = pointerOperandIndex(inst);
  std::optional<Address> access;
  for (Use &use : inst.operands()) {
    std::optional<Address> address = addressOf(use.get());
    if (!address)
      continue;
    if (use.getOperandNo() != pointerOperand)
      return Error("unsupported use of a buffer or shared-memory pointer "
                   "(Vulkan has logical addressing): " +
                   printed(inst));
    access = address;
  }
  if (!access)
    return {};
  if (!roots[access->root].choices.empty()) {
    groupAccesses.push_back({&inst, *access});
    return {};
  }
  return rewriteAccess(inst, access->root, access->index);
}

/// Points `inst`, a load, store or atomic, at element `index` of `root`, a
/// buffer or shared array.
ErrorOrSuccess KernelRewriter::rewriteAccess(Instruction &inst, unsigned root,
                                             Value *index) {
  IRBuilder<> b(&inst);
  roots[root].read |= !isa<StoreInst>(inst);
  roots[root].written |= !isa<LoadInst>(inst);
  const Root &memory = roots[root];
  if (needsAccessByParts(accessType(inst), memory.elementType))
    return accessByParts(b, inst, memory, index);
  if (splitVectorAccess(b, inst, memory, index))
    return {};
  inst.setOperand(pointerOperandIndex(inst), getPointer(b, memory, index));
  retypeAccess(b, inst, memory.elementType);
  return {};
}

/// Whether an access of `type` to memory of `elementType` elements is not an
/// access to one element or to consecutive whole elements (a vector), which
/// logical addressing cannot express directly: an aggregate, or a scalar or
/// vector element of another size.
bool KernelRewriter::needsAccessByParts(Type *type, Type *elementType) {
  if (type->isAggregateType())
    return true;
  Type *scalar = type->getScalarType();
  if (scalar->isPointerTy() || !elementType->isSingleValueType() ||
      elementType->isPointerTy())
    return false;
  return dl.getTypeSizeInBits(scalar) != dl.getTypeSizeInBits(elementType);
}

/// Loads or stores a value that does not match the memory's element type (a
/// struct in a shared array of i64, a bool in a byte buffer) as whole
/// elements: each scalar piece of the value is shifted into or out of the
/// element or elements it occupies. A store writes every element it touches
/// in full, so they must lie within the stored value (padding included):
/// logical addressing cannot reach part of an element.
ErrorOrSuccess KernelRewriter::accessByParts(IRBuilder<> &b, Instruction &inst,
                                             const Root &root, Value *index) {
  auto unsupported = [&]() {
    return Error("unsupported access to memory of another element type "
                 "(Vulkan has logical addressing): " +
                 printed(inst));
  };
  Type *element = root.elementType;
  if (!element->isIntegerTy() && !element->isFloatingPointTy())
    return unsupported();
  uint64_t slotBytes = dl.getTypeStoreSize(element);
  if (dl.getTypeSizeInBits(element) != slotBytes * 8)
    return unsupported();
  IntegerType *slotType = b.getIntNTy(static_cast<unsigned>(slotBytes * 8));
  auto pointerTo = [&](uint64_t slot) {
    Value *at = b.getInt32(static_cast<uint32_t>(slot));
    if (index)
      at = slot ? b.CreateAdd(index, at) : index;
    return getPointer(b, root, at);
  };
  // Calls `scalar(type, offset)` for each scalar piece of `type` at byte
  // `offset`, with `aggregate` producing or taking apart the pieces.
  bool ok = true;
  auto fits = [&](Type *type, uint64_t offset) {
    uint64_t bytes = dl.getTypeStoreSize(type);
    if (type->isPointerTy() || dl.getTypeSizeInBits(type) > bytes * 8)
      return false;
    if (isa<FixedVectorType>(type)) // of bools: bit-packed
      return false;
    uint64_t shift = offset % slotBytes;
    return shift + bytes <= slotBytes ||
           (shift == 0 && bytes % slotBytes == 0);
  };

  if (auto *load = dyn_cast<LoadInst>(&inst)) {
    if (!load->isSimple())
      return unsupported();
    DenseMap<uint64_t, Value *> slots;
    auto slot = [&](uint64_t i) {
      Value *&value = slots[i];
      if (!value)
        value = b.CreateBitCast(b.CreateLoad(element, pointerTo(i)), slotType);
      return value;
    };
    std::function<Value *(Type *, uint64_t)> build =
        [&](Type *type, uint64_t offset) -> Value * {
      if (auto *structType = dyn_cast<StructType>(type)) {
        const StructLayout *layout = dl.getStructLayout(structType);
        Value *value = PoisonValue::get(type);
        for (unsigned i = 0, e = structType->getNumElements(); i != e; ++i)
          value = b.CreateInsertValue(
              value,
              build(structType->getElementType(i),
                    offset + layout->getElementOffset(i)),
              i);
        return value;
      }
      if (auto *arrayType = dyn_cast<ArrayType>(type)) {
        uint64_t stride = dl.getTypeAllocSize(arrayType->getElementType());
        Value *value = PoisonValue::get(type);
        for (unsigned i = 0, e = static_cast<unsigned>(
                                 arrayType->getNumElements());
             i != e; ++i)
          value = b.CreateInsertValue(
              value, build(arrayType->getElementType(), offset + i * stride),
              i);
        return value;
      }
      if (auto *vectorType = dyn_cast<FixedVectorType>(type);
          vectorType && dl.getTypeSizeInBits(vectorType->getElementType()) ==
                             dl.getTypeStoreSize(
                                 vectorType->getElementType()) * 8) {
        Type *elementType = vectorType->getElementType();
        uint64_t stride = dl.getTypeStoreSize(elementType);
        Value *value = PoisonValue::get(type);
        for (unsigned i = 0, e = vectorType->getNumElements(); i != e; ++i)
          value = b.CreateInsertElement(
              value, build(elementType, offset + i * stride), b.getInt32(i));
        return value;
      }
      if (!fits(type, offset)) {
        ok = false;
        return PoisonValue::get(type);
      }
      uint64_t bytes = dl.getTypeStoreSize(type);
      uint64_t first = offset / slotBytes, shift = offset % slotBytes;
      Value *bits;
      if (shift + bytes <= slotBytes) {
        bits = slot(first);
        if (shift)
          bits = b.CreateLShr(bits, shift * 8);
      } else {
        IntegerType *wide = b.getIntNTy(static_cast<unsigned>(bytes * 8));
        bits = ConstantInt::get(wide, 0);
        for (uint64_t k = 0; k != bytes / slotBytes; ++k)
          bits = b.CreateOr(bits, b.CreateShl(b.CreateZExt(slot(first + k),
                                                           wide),
                                              k * slotBytes * 8));
      }
      bits = b.CreateTrunc(
          bits, b.getIntNTy(static_cast<unsigned>(dl.getTypeSizeInBits(type))));
      return type->isIntegerTy() ? bits : b.CreateBitCast(bits, type);
    };
    Value *value = build(load->getType(), 0);
    if (!ok)
      return unsupported();
    load->replaceAllUsesWith(value);
    dead.push_back(load);
    return {};
  }

  auto *store = dyn_cast<StoreInst>(&inst);
  if (!store || !store->isSimple())
    return unsupported();
  std::map<uint64_t, Value *> slots;
  auto add = [&](uint64_t i, Value *bits) {
    Value *&value = slots[i];
    value = value ? b.CreateOr(value, bits) : bits;
  };
  std::function<void(Value *, Type *, uint64_t)> scatter =
      [&](Value *value, Type *type, uint64_t offset) {
        if (auto *structType = dyn_cast<StructType>(type)) {
          const StructLayout *layout = dl.getStructLayout(structType);
          for (unsigned i = 0, e = structType->getNumElements(); i != e; ++i)
            scatter(b.CreateExtractValue(value, i),
                    structType->getElementType(i),
                    offset + layout->getElementOffset(i));
          return;
        }
        if (auto *arrayType = dyn_cast<ArrayType>(type)) {
          uint64_t stride = dl.getTypeAllocSize(arrayType->getElementType());
          for (unsigned i = 0, e = static_cast<unsigned>(
                                   arrayType->getNumElements());
               i != e; ++i)
            scatter(b.CreateExtractValue(value, i),
                    arrayType->getElementType(), offset + i * stride);
          return;
        }
        if (auto *vectorType = dyn_cast<FixedVectorType>(type);
            vectorType &&
            dl.getTypeSizeInBits(vectorType->getElementType()) ==
                dl.getTypeStoreSize(vectorType->getElementType()) * 8) {
          Type *elementType = vectorType->getElementType();
          uint64_t stride = dl.getTypeStoreSize(elementType);
          for (unsigned i = 0, e = vectorType->getNumElements(); i != e; ++i)
            scatter(b.CreateExtractElement(value, b.getInt32(i)),
                    elementType, offset + i * stride);
          return;
        }
        if (!fits(type, offset)) {
          ok = false;
          return;
        }
        uint64_t bytes = dl.getTypeStoreSize(type);
        uint64_t first = offset / slotBytes, shift = offset % slotBytes;
        Value *bits = type->isIntegerTy()
                          ? value
                          : b.CreateBitCast(
                                value, b.getIntNTy(static_cast<unsigned>(
                                           dl.getTypeSizeInBits(type))));
        if (shift + bytes <= slotBytes) {
          bits = b.CreateZExt(bits, slotType);
          if (shift)
            bits = b.CreateShl(bits, shift * 8);
          add(first, bits);
          return;
        }
        bits = b.CreateZExt(bits, b.getIntNTy(static_cast<unsigned>(bytes * 8)));
        for (uint64_t k = 0; k != bytes / slotBytes; ++k)
          add(first + k,
              b.CreateTrunc(b.CreateLShr(bits, k * slotBytes * 8), slotType));
      };
  Type *type = store->getValueOperand()->getType();
  scatter(store->getValueOperand(), type, 0);
  if (!ok || slots.empty() ||
      (slots.rbegin()->first + 1) * slotBytes > dl.getTypeAllocSize(type))
    return unsupported();
  for (auto &[i, bits] : slots)
    b.CreateStore(b.CreateBitCast(bits, element), pointerTo(i));
  dead.push_back(store);
  return {};
}

/// Splits a vector load or store into one access per element: logical SPIR-V
/// types the memory, and an access chain into an array of floats cannot yield
/// a pointer to a float vector (LLVM's SPIR-V backend emits one that Vulkan
/// rejects). Returns whether `inst` was split.
bool KernelRewriter::splitVectorAccess(IRBuilder<> &b, Instruction &inst,
                                       const Root &root, Value *index) {
  Type *accessType = nullptr;
  if (auto *load = dyn_cast<LoadInst>(&inst))
    accessType = load->getType();
  else if (auto *store = dyn_cast<StoreInst>(&inst))
    accessType = store->getValueOperand()->getType();
  auto *vectorType = dyn_cast_or_null<FixedVectorType>(accessType);
  Type *element = root.elementType;
  if (!vectorType || element->isAggregateType() ||
      vectorType->getScalarSizeInBits() != element->getPrimitiveSizeInBits())
    return false;
  unsigned count = vectorType->getNumElements();
  auto *elementVector = FixedVectorType::get(element, count);
  auto pointerTo = [&](unsigned i) {
    Value *at = b.getInt32(i);
    if (index)
      at = i ? b.CreateAdd(index, at) : index;
    return getPointer(b, root, at);
  };
  if (auto *load = dyn_cast<LoadInst>(&inst)) {
    Value *vector = PoisonValue::get(elementVector);
    for (unsigned i = 0; i != count; ++i)
      vector = b.CreateInsertElement(
          vector, b.CreateLoad(element, pointerTo(i)), b.getInt32(i));
    if (elementVector != vectorType)
      vector = b.CreateBitCast(vector, vectorType);
    load->replaceAllUsesWith(vector);
  } else {
    Value *value = cast<StoreInst>(inst).getValueOperand();
    if (elementVector != vectorType)
      value = b.CreateBitCast(value, elementVector);
    for (unsigned i = 0; i != count; ++i)
      b.CreateStore(b.CreateExtractElement(value, b.getInt32(i)),
                    pointerTo(i));
  }
  dead.push_back(&inst);
  return true;
}

/// Makes a load or store of a same-sized type other than the memory's element
/// type (e.g. an i32 store of float bits into a float array, which InstCombine
/// forms by folding a bitcast) access the element type, with a bitcast:
/// logical SPIR-V types the memory, and an access chain to a float cannot be
/// read as an integer.
void KernelRewriter::retypeAccess(IRBuilder<> &b, Instruction &inst,
                                  Type *elementType) {
  auto asElements = [&](Type *type) -> Type * {
    Type *scalar = type->getScalarType();
    if (scalar == elementType || !scalar->isSingleValueType() ||
        scalar->isPointerTy() ||
        scalar->getPrimitiveSizeInBits() !=
            elementType->getPrimitiveSizeInBits())
      return nullptr;
    if (auto *vector = dyn_cast<FixedVectorType>(type))
      return FixedVectorType::get(elementType, vector->getNumElements());
    return elementType;
  };
  if (auto *store = dyn_cast<StoreInst>(&inst)) {
    Value *value = store->getValueOperand();
    if (Type *type = asElements(value->getType()))
      store->setOperand(0, b.CreateBitCast(value, type));
    return;
  }
  auto *load = dyn_cast<LoadInst>(&inst);
  if (!load)
    return;
  Type *type = asElements(load->getType());
  if (!type)
    return;
  IRBuilder<> after(load->getNextNode());
  LoadInst *typed = after.CreateAlignedLoad(type, load->getPointerOperand(),
                                            load->getAlign(),
                                            load->isVolatile());
  Value *cast = after.CreateBitCast(typed, load->getType());
  load->replaceAllUsesWith(cast);
  dead.push_back(load);
}

ErrorOrSuccess KernelRewriter::finishPhis() {
  for (auto [phi, index, selector] : phis) {
    unsigned group = addresses.lookup(phi).root;
    // A block may reach the phi along several edges, all with one value.
    DenseMap<BasicBlock *, Value *> selectors;
    for (unsigned i = 0, e = phi->getNumIncomingValues(); i != e; ++i) {
      Value *value = phi->getIncomingValue(i);
      BasicBlock *block = phi->getIncomingBlock(i);
      Value *incoming;
      Value *choice = ConstantInt::get(i32, 0);
      if (isa<UndefValue>(value)) {
        incoming = PoisonValue::get(i32);
      } else if (auto it = addresses.find(value); it != addresses.end()) {
        incoming = it->second.index ? it->second.index
                                    : ConstantInt::get(i32, 0);
        if (selector) {
          Value *&cached = selectors[block];
          if (!cached) {
            IRBuilder<> b(block->getTerminator());
            cached = selectorIn(b, it->second, group);
          }
          choice = cached;
        }
      } else {
        return Error("phi mixes buffer or shared-memory pointers with other "
                     "pointers (Vulkan has logical addressing): " +
                     printed(*phi));
      }
      index->addIncoming(incoming, block);
      if (selector)
        selector->addIncoming(choice, block);
    }
  }
  return {};
}

/// Expands each load, store or atomic through a pointer into one of several
/// buffers or shared arrays (a group address): logical addressing cannot pick
/// between storage buffers at run time. A load reads every choice, the ones
/// not selected at element 0, and selects the value, which keeps the control
/// flow as it was (LLVM's SPIR-V structurizer mishandles some added
/// branches); a store or atomic branches on the selector, one path per choice.
ErrorOrSuccess KernelRewriter::expandGroupAccesses() {
  for (auto [inst, address] : groupAccesses) {
    ArrayRef<unsigned> choices = roots[address.root].choices;
    SmallVector<std::pair<Instruction *, unsigned>> copies;
    SmallVector<Value *> indices;
    if (auto *load = dyn_cast<LoadInst>(inst); load && load->isSimple()) {
      IRBuilder<> b(inst);
      SmallVector<Value *> isChoice;
      for (auto [i, choice] : enumerate(choices)) {
        isChoice.push_back(b.CreateICmpEQ(
            address.selector, b.getInt32(static_cast<uint32_t>(i))));
        indices.push_back(address.index
                              ? b.CreateSelect(isChoice.back(), address.index,
                                               b.getInt32(0))
                              : nullptr);
        Instruction *copy = inst->clone();
        copy->insertBefore(inst->getIterator());
        copies.push_back({copy, choice});
      }
      Value *value = copies.back().first;
      for (size_t i = copies.size() - 1; i-- > 0;)
        value = b.CreateSelect(isChoice[i], copies[i].first, value);
      inst->replaceAllUsesWith(value);
    } else {
      BasicBlock *head = inst->getParent();
      BasicBlock *join = head->splitBasicBlock(inst->getIterator(), "vk.join");
      head->getTerminator()->eraseFromParent();
      Function *fn = head->getParent();
      PHINode *result = nullptr;
      if (!inst->getType()->isVoidTy()) {
        result = PHINode::Create(inst->getType(), choices.size(), "",
                                 join->begin());
        inst->replaceAllUsesWith(result);
      }
      SmallVector<BasicBlock *> paths;
      for (unsigned choice : choices) {
        BasicBlock *path =
            BasicBlock::Create(ctx, "vk.root" + Twine(choice), fn, join);
        Instruction *copy = inst->clone();
        copy->insertInto(path, path->end());
        UncondBrInst::Create(join, path);
        if (result)
          result->addIncoming(copy, path);
        paths.push_back(path);
        copies.push_back({copy, choice});
        indices.push_back(address.index);
      }
      // A chain of tests; the last choice is the remaining case.
      BasicBlock *test = head;
      for (size_t i = 0; i + 1 < paths.size(); ++i) {
        BasicBlock *otherwise =
            i + 2 == paths.size()
                ? paths.back()
                : BasicBlock::Create(ctx, "vk.test", fn, paths[i + 1]);
        IRBuilder<> b(test);
        b.CreateCondBr(b.CreateICmpEQ(address.selector,
                                      b.getInt32(static_cast<uint32_t>(i))),
                       paths[i], otherwise);
        test = otherwise;
      }
    }
    for (auto [i, copy] : enumerate(copies))
      if (ErrorOrSuccess err =
              rewriteAccess(*copy.first, copy.second, indices[i]))
        return err;
    dead.push_back(inst);
  }
  return {};
}

/// Expands each memset or memcpy on a buffer or shared array (Mojo's idiom
/// for zeroing or copying a tile) into a loop over its elements: logical
/// addressing has no byte-wise copy.
ErrorOrSuccess KernelRewriter::expandMemIntrinsics() {
  for (MemIntrinsic *memory : memIntrinsics) {
    auto unsupported = [&]() {
      return Error("unsupported memset or memcpy on a buffer or shared "
                   "array: " +
                   printed(*memory));
    };
    auto destination = addresses.find(memory->getDest());
    if (destination == addresses.end() ||
        !roots[destination->second.root].choices.empty() ||
        memory->isVolatile())
      return unsupported();
    Address dest = destination->second;
    roots[dest.root].written = true;
    const Root &root = roots[dest.root];
    Type *element = root.elementType;
    if (!element->isIntegerTy() && !element->isFloatingPointTy())
      return unsupported();
    uint64_t size = dl.getTypeStoreSize(element);
    std::optional<Address> source;
    if (auto *transfer = dyn_cast<MemTransferInst>(memory)) {
      auto it = addresses.find(transfer->getSource());
      if (it == addresses.end() || !roots[it->second.root].choices.empty() ||
          roots[it->second.root].elementType != element)
        return unsupported();
      source = it->second;
      roots[source->root].read = true;
    } else if (!isa<MemSetInst>(memory)) {
      return unsupported();
    }

    IRBuilder<> b(memory);
    Value *count;
    if (auto *length = dyn_cast<ConstantInt>(memory->getLength())) {
      if (length->getZExtValue() % size)
        return unsupported();
      count = b.getInt32(static_cast<uint32_t>(length->getZExtValue() / size));
    } else {
      count = b.CreateTrunc(
          b.CreateUDiv(memory->getLength(),
                       ConstantInt::get(memory->getLength()->getType(), size)),
          i32);
    }
    // The element a memset stores: its byte in every byte of the element.
    Value *fill = nullptr;
    if (auto *set = dyn_cast<MemSetInst>(memory)) {
      IntegerType *bits = b.getIntNTy(static_cast<unsigned>(size * 8));
      Value *byte = b.CreateZExt(set->getValue(), bits);
      fill = b.CreateMul(byte, ConstantInt::get(bits, APInt::getSplat(
                                                          bits->getBitWidth(),
                                                          APInt(8, 1))));
      fill = b.CreateBitCast(fill, element);
    }

    BasicBlock *head = memory->getParent();
    BasicBlock *exit = head->splitBasicBlock(memory->getIterator(), "vk.mem.exit");
    head->getTerminator()->eraseFromParent();
    BasicBlock *loop =
        BasicBlock::Create(ctx, "vk.mem", head->getParent(), exit);
    IRBuilder<> hb(head);
    hb.CreateCondBr(hb.CreateICmpEQ(count, hb.getInt32(0)), exit, loop);
    IRBuilder<> lb(loop);
    PHINode *i = lb.CreatePHI(i32, 2);
    i->addIncoming(lb.getInt32(0), head);
    auto at = [&](const Address &address) {
      return address.index ? lb.CreateAdd(address.index, i) : i;
    };
    Value *value =
        source ? lb.CreateLoad(element,
                               getPointer(lb, roots[source->root], at(*source)))
               : fill;
    lb.CreateStore(value, getPointer(lb, root, at(dest)));
    Value *next = lb.CreateAdd(i, lb.getInt32(1));
    i->addIncoming(next, loop);
    lb.CreateCondBr(lb.CreateICmpULT(next, count), loop, exit);
    dead.push_back(memory);
  }
  return {};
}

/// Erases the rewritten pointer computations. They may form cycles through
/// phis, so every user must itself be rewritten before any is erased.
ErrorOrSuccess KernelRewriter::eraseDead() {
  SmallPtrSet<Instruction *, 32> deadSet(dead.begin(), dead.end());
  for (Instruction *inst : dead)
    for (User *user : inst->users())
      if (!deadSet.contains(cast<Instruction>(user)))
        return Error("rewritten pointer still in use: " + printed(*user));
  for (Instruction *inst : dead)
    inst->dropAllReferences();
  for (Instruction *inst : dead)
    inst->eraseFromParent();
  return {};
}

ErrorOrSuccess KernelRewriter::run() {
  Function *entry = Function::Create(
      FunctionType::get(Type::getVoidTy(ctx), /*isVarArg=*/false),
      GlobalValue::ExternalLinkage, entryName, module);
  entry->addFnAttr("hlsl.shader", "compute");
  entry->addFnAttr("hlsl.numthreads", (Twine(localSize[0]) + "," +
                                       Twine(localSize[1]) + "," +
                                       Twine(localSize[2]))
                                          .str());
  entry->splice(entry->begin(), &kernel);
  BasicBlock &entryBlock = entry->getEntryBlock();
  IRBuilder<> b(&entryBlock, entryBlock.getFirstNonPHIOrDbgOrAlloca());

  // The arguments, flattened in order: each pointer (a pointer argument, or a
  // pointer field of an aggregate passed by value, such as a tensor) is the
  // next storage buffer; each scalar (or scalar field) is the next 64-bit slot
  // of the params buffer, bound after the last pointer.
  SmallVector<Leaf> leaves;
  for (Argument &arg : kernel.args()) {
    if (ErrorOrSuccess err = flattenArgument(arg, leaves))
      return err;
  }
  // Argument sizes, for the description.
  json::Array argInfo;
  for (Argument &arg : kernel.args()) {
    Type *type = arg.hasByValAttr() ? arg.getParamByValType() : arg.getType();
    argInfo.push_back(json::Object{
        {"bytes", static_cast<int64_t>(dl.getTypeAllocSize(type))},
        {"byval", arg.hasByValAttr()}});
  }
  unsigned binding = 0;
  SmallVector<const Leaf *> scalars;
  SmallVector<std::pair<const Leaf *, unsigned>> buffers; // leaf, root
  for (const Leaf &leaf : leaves) {
    if (!leaf.type->isPointerTy()) {
      scalars.push_back(&leaf);
      continue;
    }
    Type *elementType = nullptr;
    for (Value *value : leaf.values) {
      Type *type = inferElementType(value);
      if (!elementType || (elementType->isIntegerTy(8) && !type->isIntegerTy(8)))
        elementType = type;
    }
    if (!elementType)
      elementType = Type::getFloatTy(ctx);
    Type *handleType = TargetExtType::get(
        ctx, "spirv.VulkanBuffer", {ArrayType::get(elementType, 0)},
        {kStorageBufferClass, /*writable=*/1});
    Value *handle =
        createHandle(b, handleType, binding, ("buf" + Twine(binding)).str());
    roots.push_back({handle, nullptr, elementType, handleType, {}});
    buffers.push_back({&leaf, static_cast<unsigned>(roots.size() - 1)});
    for (Value *value : leaf.values)
      addresses[value] = {static_cast<unsigned>(roots.size() - 1), nullptr};
    ++binding;
  }
  // With a dynamic workgroup size, the block size follows the scalars.
  std::optional<unsigned> paramsBinding;
  if (!scalars.empty() || dynamicSize) {
    paramsBinding = binding;
    Type *i64 = Type::getInt64Ty(ctx);
    Type *handleType = TargetExtType::get(ctx, "spirv.VulkanBuffer",
                                          {ArrayType::get(i64, 0)},
                                          {kStorageBufferClass, /*writable=*/1});
    Root params{createHandle(b, handleType, binding, "params"), nullptr, i64,
                handleType, {}};
    for (auto [slot, leaf] : enumerate(scalars)) {
      Value *bits = b.CreateLoad(
          i64, getPointer(b, params, b.getInt32(static_cast<uint32_t>(slot))));
      Type *type = leaf->type;
      uint64_t size = dl.getTypeSizeInBits(type);
      if (size == 0 || size > 64 || type->isAggregateType())
        return Error("unsupported kernel argument type in @" +
                     kernel.getName());
      Value *value = b.CreateTrunc(bits, b.getIntNTy(size));
      if (!type->isIntegerTy())
        value = b.CreateBitCast(value, type);
      // A field read from memory may be loaded as another type of its size.
      for (Value *use : leaf->values)
        use->replaceAllUsesWith(use->getType() == type
                                    ? value
                                    : b.CreateBitCast(value, use->getType()));
    }
    if (dynamicSize)
      for (unsigned d = 0; d != 3; ++d)
        blockDims[d] = b.CreateTrunc(
            b.CreateLoad(i64, getPointer(b, params,
                                         b.getInt32(static_cast<uint32_t>(
                                             scalars.size() + d)))),
            i32);
  }

  if (ErrorOrSuccess err = addSharedRoots(*entry, b))
    return err;

  ReversePostOrderTraversal<Function *> rpot(entry);
  for (BasicBlock *block : rpot)
    for (Instruction &inst : make_early_inc_range(*block))
      if (ErrorOrSuccess err = visit(inst))
        return err;
  if (ErrorOrSuccess err = finishPhis())
    return err;
  if (ErrorOrSuccess err = expandGroupAccesses())
    return err;
  if (ErrorOrSuccess err = expandMemIntrinsics())
    return err;
  if (ErrorOrSuccess err = eraseDead())
    return err;

  for (Argument &arg : kernel.args())
    if (!arg.use_empty())
      return Error("kernel argument still in use: " + printed(arg));
  std::string kernelName = kernel.getName().str();
  kernel.eraseFromParent();
  for (GlobalVariable *global : retypedGlobals)
    if (global->use_empty())
      global->eraseFromParent();

  for (Instruction &inst : instructions(*entry))
    if (auto *call = dyn_cast<CallBase>(&inst);
        call && !isa<IntrinsicInst>(call))
      return Error("kernel still calls a function (Vulkan entry points must "
                   "be fully inlined): " +
                   printed(inst));

  // The launch description: where each argument's pointers and scalars go.
  json::Array bufferInfo;
  for (auto [i, buffer] : enumerate(buffers)) {
    const Root &root = roots[buffer.second];
    bufferInfo.push_back(json::Object{
        {"binding", static_cast<int64_t>(i)},
        {"arg", static_cast<int64_t>(buffer.first->arg)},
        {"offset", static_cast<int64_t>(buffer.first->offset)},
        {"element", printed(*root.elementType)},
        {"access", root.read && root.written ? "read_write"
                   : root.written            ? "write"
                   : root.read               ? "read"
                                             : "none"}});
  }
  json::Value paramsInfo = nullptr;
  if (paramsBinding) {
    json::Array slots;
    for (const Leaf *leaf : scalars)
      slots.push_back(json::Object{{"arg", static_cast<int64_t>(leaf->arg)},
                                   {"offset", static_cast<int64_t>(leaf->offset)},
                                   {"type", printed(*leaf->type)}});
    if (dynamicSize)
      for (const char *dim : {"block_dim.x", "block_dim.y", "block_dim.z"})
        slots.push_back(json::Object{{"value", dim}});
    paramsInfo = json::Object{{"binding", static_cast<int64_t>(*paramsBinding)},
                              {"slots", std::move(slots)}};
  }
  json::Array size{static_cast<int64_t>(localSize[0]),
                   static_cast<int64_t>(localSize[1]),
                   static_cast<int64_t>(localSize[2])};
  json::Object workgroupSize =
      dynamicSize ? json::Object{{"default", std::move(size)},
                                 {"spec_ids", json::Array{0, 1, 2}}}
                  : json::Object{{"fixed", std::move(size)}};
  raw_string_ostream os(description);
  os << json::Value(json::Object{{"abi", "mojo-vulkan-1"},
                                 {"entry", entryName},
                                 {"kernel", kernelName},
                                 {"workgroup_size", std::move(workgroupSize)},
                                 {"args", std::move(argInfo)},
                                 {"buffers", std::move(bufferInfo)},
                                 {"params", std::move(paramsInfo)}});
  return {};
}

/// Inlines every call to a function with a body into `kernel`: a Vulkan entry
/// point cannot pass its buffers to other functions. Mojo keeps cold paths,
/// such as a failed `debug_assert`, out of line. Calls left after a few
/// rounds (recursion) are reported after the rewrite.
ErrorOrSuccess inlineCalls(Function &kernel) {
  for (unsigned round = 0; round != 8; ++round) {
    SmallVector<CallBase *> calls;
    for (Instruction &inst : instructions(kernel))
      if (auto *call = dyn_cast<CallBase>(&inst))
        if (Function *callee = call->getCalledFunction();
            callee && !callee->isDeclaration() && callee != &kernel)
          calls.push_back(call);
    if (calls.empty())
      return {};
    for (CallBase *call : calls) {
      InlineFunctionInfo info;
      if (InlineResult result = InlineFunction(*call, info);
          !result.isSuccess())
        return Error("cannot inline into a Vulkan entry point (" +
                     Twine(result.getFailureReason()) + "): " + printed(*call));
    }
  }
  return {};
}

/// Makes each trap (a failed `debug_assert`) end the invocation: a compute
/// shader cannot trap, and reaching the OpUnreachable after it is undefined.
void trapsToReturns(Function &kernel) {
  if (!kernel.getReturnType()->isVoidTy())
    return;
  SmallVector<IntrinsicInst *> traps;
  for (Instruction &inst : instructions(kernel))
    if (auto *intrinsic = dyn_cast<IntrinsicInst>(&inst))
      if (intrinsic->getIntrinsicID() == Intrinsic::trap ||
          intrinsic->getIntrinsicID() == Intrinsic::ubsantrap)
        traps.push_back(intrinsic);
  for (IntrinsicInst *trap : traps) {
    BasicBlock *block = trap->getParent();
    changeToUnreachable(trap->getNextNode());
    Instruction *unreachable = block->getTerminator();
    ReturnInst::Create(kernel.getContext(), nullptr, unreachable->getIterator());
    unreachable->eraseFromParent();
    trap->eraseFromParent();
  }
}

/// Whether `fn` computes on vectors of more than four elements, which Vulkan
/// SPIR-V cannot hold: Mojo's SIMD width is 128 bits, so a kernel on f16
/// data works on <8 x half>.
bool hasWideVectors(Function &fn) {
  auto wide = [](Type *type) {
    auto *vector = dyn_cast<FixedVectorType>(type);
    return vector && vector->getNumElements() > 4;
  };
  for (Instruction &inst : instructions(fn))
    if (wide(inst.getType()) || any_of(inst.operands(), [&](Value *operand) {
          return wide(operand->getType());
        }))
      return true;
  return false;
}

/// Prepares `fn`, an entry point, for LLVM's SPIR-V codegen: splits wide
/// vector operations into scalars, and structures the control flow. Vulkan
/// requires structured control flow (every branch and loop with a merge
/// block), which LLVM's SPIR-V structurizer derives, but it expects the
/// nearly structured code of HLSL: on the jump-threaded and sunk CFGs of
/// Mojo's optimized kernels it can misplace merge blocks. LLVM's
/// StructurizeCFG (as for AMDGPU) leaves it nothing to restructure.
void prepareForCodegen(Function &fn) {
  LoopAnalysisManager loops;
  FunctionAnalysisManager functions;
  CGSCCAnalysisManager sccs;
  ModuleAnalysisManager modules;
  PassBuilder builder;
  builder.registerModuleAnalyses(modules);
  builder.registerCGSCCAnalyses(sccs);
  builder.registerFunctionAnalyses(functions);
  builder.registerLoopAnalyses(loops);
  builder.crossRegisterProxies(loops, functions, sccs, modules);
  FunctionPassManager passes;
  if (hasWideVectors(fn)) {
    ScalarizerPassOptions options;
    options.ScalarizeLoadStore = true;
    passes.addPass(ScalarizerPass(options));
  }
  passes.addPass(LowerSwitchPass());
  passes.addPass(UnifyLoopExitsPass());
  passes.addPass(FixIrreduciblePass());
  passes.addPass(StructurizeCFGPass());
  passes.run(fn, functions);
}

} // namespace

ErrorOrSuccess lowerSPIRVKernelsToVulkan(Module &module,
                                         VulkanModuleInfo &info) {
  ErrorOr<std::pair<LocalSize, bool>> localSize = getLocalSize();
  if (localSize.isError())
    return localSize.takeError();
  // Without MOJO_SPIRV_LOCAL_SIZE, the workgroup size is each launch's block
  // size: MAX picks it at the launch, while Vulkan fixes it per pipeline.
  info.specializeWorkgroupSize = !localSize->second;

  SmallVector<Function *> kernels;
  for (Function &fn : module)
    if (!fn.isDeclaration() && fn.hasFnAttribute(kKernelAttr))
      kernels.push_back(&fn);
  for (Function *kernel : kernels) {
    std::string name =
        kernels.size() == 1 ? std::string("main") : kernel->getName().str();
    if (kernels.size() != 1)
      kernel->setName(name + ".kernel");
    // A kernel declaring its block size (MAX_THREADS_PER_BLOCK_METADATA) gets
    // it as its workgroup size; others take MOJO_SPIRV_LOCAL_SIZE.
    LocalSize size = localSize->first;
    if (Attribute attr = kernel->getFnAttribute("mojo.spirv.numthreads");
        attr.isStringAttribute()) {
      unsigned threads = 0;
      if (attr.getValueAsString().getAsInteger(10, threads) || threads == 0)
        return Error("invalid mojo.spirv.numthreads on @" + kernel->getName());
      size = {threads, 1, 1};
    }
    if (ErrorOrSuccess err = inlineCalls(*kernel))
      return err;
    trapsToReturns(*kernel);
    KernelRewriter rewriter(module, *kernel, name, size,
                            info.specializeWorkgroupSize);
    if (ErrorOrSuccess err = rewriter.run())
      return err;
    info.descriptions.push_back(rewriter.getDescription());
  }
  for (Function &fn : module)
    if (fn.hasFnAttribute("hlsl.shader"))
      prepareForCodegen(fn);

  // Helpers left behind after inlining are dead: Vulkan has no calls between
  // entry points and functions taking raw pointers.
  for (Function &fn : make_early_inc_range(module))
    if (!fn.isDeclaration() && fn.use_empty() &&
        !fn.hasFnAttribute("hlsl.shader"))
      fn.eraseFromParent();

  std::string message;
  raw_string_ostream os(message);
  if (verifyModule(module, &os))
    return Error("invalid module after the Vulkan kernel rewrite: " +
                 Twine(message));
  return {};
}


ErrorOr<std::vector<uint32_t>>
finishVulkanSPIRV(ArrayRef<uint32_t> words, const VulkanModuleInfo &info) {
  // SPIR-V opcodes and operands used below.
  enum : uint32_t {
    OpSourceContinued = 2, OpSource = 3, OpSourceExtension = 4, OpName = 5,
    OpMemberName = 6, OpString = 7, OpExtension = 10, OpExtInstImport = 11,
    OpMemoryModel = 14, OpEntryPoint = 15, OpExecutionMode = 16,
    OpCapability = 17, OpTypeInt = 21, OpTypeVector = 23,
    OpSpecConstant = 50, OpSpecConstantComposite = 51, OpFunction = 54,
    OpDecorate = 71, OpModuleProcessed = 330, OpExecutionModeId = 331,
    DecorationSpecId = 1, DecorationBuiltIn = 11, BuiltInWorkgroupSize = 25,
    ExecutionModeLocalSize = 17,
  };
  if (words.size() < 5 || words[0] != 0x07230203)
    return Error("not a SPIR-V module");
  uint32_t bound = words[3];
  // The end of the debug section (OpModuleProcessed goes last in it, and
  // annotations follow), the first function, existing u32 and v3u32 types,
  // and the shader's LocalSize.
  size_t debugEnd = 0, firstFunction = 0;
  uint32_t u32 = 0, v3u32 = 0;
  std::array<uint32_t, 3> localSize = {1, 1, 1};
  for (size_t at = 5; at < words.size();) {
    uint32_t count = words[at] >> 16, opcode = words[at] & 0xffff;
    if (count == 0 || at + count > words.size())
      return Error("malformed SPIR-V module");
    switch (opcode) {
    case OpCapability: case OpExtension: case OpExtInstImport:
    case OpMemoryModel: case OpEntryPoint: case OpExecutionModeId:
    case OpString: case OpSourceExtension: case OpSource:
    case OpSourceContinued: case OpName: case OpMemberName:
    case OpModuleProcessed:
      break;
    case OpExecutionMode:
      if (count == 6 && words[at + 2] == ExecutionModeLocalSize)
        localSize = {words[at + 3], words[at + 4], words[at + 5]};
      break;
    default:
      if (!debugEnd)
        debugEnd = at;
      if (opcode == OpTypeInt && count == 4 && words[at + 2] == 32 &&
          words[at + 3] == 0)
        u32 = words[at + 1];
      if (opcode == OpTypeVector && count == 4 && u32 &&
          words[at + 2] == u32 && words[at + 3] == 3)
        v3u32 = words[at + 1];
      if (opcode == OpFunction && !firstFunction)
        firstFunction = at;
      break;
    }
    at += count;
  }
  if (!debugEnd || !firstFunction)
    return Error("SPIR-V module without a function");

  // OpModuleProcessed "mojo.vulkan.abi <json>" per kernel: drivers ignore
  // debug instructions, and a runtime finds each kernel's launch description
  // inside its module.
  std::vector<uint32_t> debug;
  for (const std::string &description : info.descriptions) {
    std::string text = "mojo.vulkan.abi " + description;
    size_t stringWords = text.size() / 4 + 1;  // with the terminating zero
    if (stringWords + 1 > 0xffff)
      return Error("kernel description too long for SPIR-V");
    debug.push_back(static_cast<uint32_t>((stringWords + 1) << 16 |
                                          OpModuleProcessed));
    size_t first = debug.size();
    debug.resize(first + stringWords, 0);
    std::memcpy(&debug[first], text.data(), text.size());
  }

  // The workgroup size as specialization constants 0, 1 and 2 (defaulting to
  // the shader's LocalSize), which a WorkgroupSize built-in overrides.
  std::vector<uint32_t> annotations, constants;
  if (info.specializeWorkgroupSize) {
    if (!u32) {
      u32 = bound++;
      constants.insert(constants.end(), {4u << 16 | OpTypeInt, u32, 32, 0});
    }
    if (!v3u32) {
      v3u32 = bound++;
      constants.insert(constants.end(), {4u << 16 | OpTypeVector, v3u32, u32, 3});
    }
    std::array<uint32_t, 3> ids;
    for (unsigned d = 0; d != 3; ++d) {
      ids[d] = bound++;
      constants.insert(constants.end(),
                       {4u << 16 | OpSpecConstant, u32, ids[d], localSize[d]});
      annotations.insert(annotations.end(),
                         {4u << 16 | OpDecorate, ids[d], DecorationSpecId, d});
    }
    uint32_t workgroupSize = bound++;
    constants.insert(constants.end(),
                     {6u << 16 | OpSpecConstantComposite, v3u32, workgroupSize,
                      ids[0], ids[1], ids[2]});
    annotations.insert(annotations.end(),
                       {4u << 16 | OpDecorate, workgroupSize, DecorationBuiltIn,
                        BuiltInWorkgroupSize});
  }

  std::vector<uint32_t> result(words.begin(), words.begin() + debugEnd);
  result[3] = bound;
  result.insert(result.end(), debug.begin(), debug.end());
  result.insert(result.end(), annotations.begin(), annotations.end());
  result.insert(result.end(), words.begin() + debugEnd,
                words.begin() + firstFunction);
  result.insert(result.end(), constants.begin(), constants.end());
  result.insert(result.end(), words.begin() + firstFunction, words.end());
  return result;
}

} // namespace M::KGEN
