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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <functional>
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

/// Reads the workgroup size from `MOJO_SPIRV_LOCAL_SIZE` ("x[,y[,z]]").
ErrorOr<LocalSize> getLocalSize() {
  LocalSize size = {64, 1, 1};
  std::optional<std::string> env =
      sys::Process::GetEnv("MOJO_SPIRV_LOCAL_SIZE");
  if (!env)
    return size;
  SmallVector<StringRef> parts;
  StringRef(*env).split(parts, ',');
  if (parts.size() > 3)
    return Error("invalid MOJO_SPIRV_LOCAL_SIZE '" + Twine(*env) + "'");
  for (auto [i, part] : enumerate(parts))
    if (part.trim().getAsInteger(10, size[i]) || size[i] == 0)
      return Error("invalid MOJO_SPIRV_LOCAL_SIZE '" + Twine(*env) + "'");
  return size;
}

std::string printed(const Value &value) {
  std::string text;
  raw_string_ostream os(text);
  value.print(os);
  return StringRef(text).trim().str();
}

/// Returns the scalar type that memory reached through `root` is accessed as:
/// the element type of the loads, stores and atomics on it or on the GEPs and
/// casts based on it. A byte type (from byte-offset GEPs and memcpy-like
/// accesses) loses to any other; float if the memory is never accessed.
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
      else if (isa<GetElementPtrInst, AddrSpaceCastInst, BitCastInst>(user))
        work.push_back(user);
      if (!type)
        continue;
      type = type->getScalarType();
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
                 LocalSize localSize)
      : module(module), kernel(kernel), entryName(entryName),
        localSize(localSize), dl(module.getDataLayout()),
        ctx(module.getContext()), i32(Type::getInt32Ty(ctx)) {}

  ErrorOrSuccess run();

private:
  /// Memory a pointer reaches: a storage buffer or a shared array.
  struct Root {
    Value *handle = nullptr;           // the buffer's resource handle, or
    GlobalVariable *shared = nullptr;  // the Workgroup array
    Type *elementType = nullptr;
    Type *storageType = nullptr;       // handle or array type
  };
  /// A pointer as (root, i32 element index); a null index means element 0.
  struct Address {
    unsigned root;
    Value *index;
  };
  /// One flattened piece of an argument: the argument itself, or a pointer or
  /// scalar field of an aggregate one, with the values that read it.
  struct Leaf {
    Type *type;
    SmallVector<Value *> values;
  };

  ErrorOrSuccess flattenArgument(Argument &arg, SmallVectorImpl<Leaf> &leaves);

  Value *createHandle(IRBuilder<> &b, Type *handleType, unsigned binding,
                      StringRef name);
  Value *getPointer(IRBuilder<> &b, const Root &root, Value *index);
  ErrorOrSuccess addSharedRoots(Function &entry, IRBuilder<> &b);
  ErrorOr<Value *> elementIndex(IRBuilder<> &b, GetElementPtrInst &gep,
                                Type *elementType);
  Value *scaledIndex(IRBuilder<> &b, Value *value, int64_t bytes,
                     int64_t size);
  std::optional<unsigned> rootOf(Value *value,
                                 SmallPtrSetImpl<Value *> &visiting);
  ErrorOrSuccess visit(Instruction &inst);
  void retypeAccess(IRBuilder<> &b, Instruction &inst, Type *elementType);
  bool splitVectorAccess(IRBuilder<> &b, Instruction &inst, const Root &root,
                         Value *index);
  ErrorOrSuccess finishPhis();
  ErrorOrSuccess eraseDead();

  Module &module;
  Function &kernel;
  StringRef entryName;
  LocalSize localSize;
  const DataLayout &dl;
  LLVMContext &ctx;
  Type *i32;
  SmallVector<Root> roots;
  DenseMap<Value *, Address> addresses;
  /// Pointer phis and the index phis that replace them, filled in once every
  /// incoming address (some along back edges) is known.
  SmallVector<std::pair<PHINode *, PHINode *>> phis;
  SmallVector<Instruction *> dead;
  SmallVector<GlobalVariable *> retypedGlobals;
};

/// Appends `arg`'s leaves to `leaves`: the argument itself, or each field of
/// an aggregate (struct or array, recursively) in field order. An aggregate is
/// only read field by field, through `extractvalue`s, which become the leaves'
/// values.
ErrorOrSuccess KernelRewriter::flattenArgument(Argument &arg,
                                               SmallVectorImpl<Leaf> &leaves) {
  Type *type = arg.getType();
  if (!type->isAggregateType()) {
    leaves.push_back({type, {&arg}});
    return {};
  }
  SmallVector<std::pair<SmallVector<unsigned>, Type *>> fields;
  SmallVector<unsigned> path;
  std::function<void(Type *)> collect = [&](Type *field) {
    if (auto *structType = dyn_cast<StructType>(field)) {
      for (unsigned i = 0, e = structType->getNumElements(); i != e; ++i) {
        path.push_back(i);
        collect(structType->getElementType(i));
        path.pop_back();
      }
    } else if (auto *arrayType = dyn_cast<ArrayType>(field)) {
      for (uint64_t i = 0, e = arrayType->getNumElements(); i != e; ++i) {
        path.push_back(static_cast<unsigned>(i));
        collect(arrayType->getElementType());
        path.pop_back();
      }
    } else {
      fields.push_back({path, field});
    }
  };
  collect(type);
  size_t first = leaves.size();
  for (auto &field : fields)
    leaves.push_back({field.second, {}});

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
        return ArrayRef<unsigned>(field.first) == ArrayRef<unsigned>(fieldPath);
      });
      if (it == fields.end())
        return Error("unknown field of an aggregate kernel argument: " +
                     printed(*extract));
      leaves[first + (it - fields.begin())].values.push_back(extract);
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
        {nullptr, global, elementType, ArrayType::get(elementType, count)});
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

/// Returns the root that `value`, a pointer, addresses, following GEPs, casts,
/// phis and selects (whose inputs must agree), before they are visited.
std::optional<unsigned>
KernelRewriter::rootOf(Value *value, SmallPtrSetImpl<Value *> &visiting) {
  if (auto it = addresses.find(value); it != addresses.end())
    return it->second.root;
  if (!value->getType()->isPointerTy() || !visiting.insert(value).second)
    return std::nullopt;
  if (auto *gep = dyn_cast<GEPOperator>(value))
    return rootOf(gep->getPointerOperand(), visiting);
  if (isa<AddrSpaceCastInst, BitCastInst>(value))
    return rootOf(cast<Instruction>(value)->getOperand(0), visiting);
  SmallVector<Value *> inputs;
  if (auto *phi = dyn_cast<PHINode>(value))
    inputs.append(phi->incoming_values().begin(),
                  phi->incoming_values().end());
  else if (auto *select = dyn_cast<SelectInst>(value))
    inputs = {select->getTrueValue(), select->getFalseValue()};
  std::optional<unsigned> root;
  for (Value *input : inputs) {
    if (isa<UndefValue>(input))
      continue;
    std::optional<unsigned> inputRoot = rootOf(input, visiting);
    if (!inputRoot)
      continue; // a cycle back to `value`, or not an address
    if (root && *root != *inputRoot)
      return std::nullopt;
    root = inputRoot;
  }
  return root;
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
  // select of element indices.
  if (auto *phi = dyn_cast<PHINode>(&inst); phi && phi->getType()->isPointerTy()) {
    SmallPtrSet<Value *, 8> visiting;
    std::optional<unsigned> root = rootOf(phi, visiting);
    if (!root)
      return {};
    PHINode *index =
        PHINode::Create(i32, phi->getNumIncomingValues(), "", phi->getIterator());
    addresses[phi] = {*root, index};
    phis.push_back({phi, index});
    dead.push_back(phi);
    return {};
  }
  if (auto *select = dyn_cast<SelectInst>(&inst);
      select && select->getType()->isPointerTy()) {
    std::optional<Address> t = addressOf(select->getTrueValue());
    std::optional<Address> f = addressOf(select->getFalseValue());
    if (!t && !f)
      return {};
    if (!t || !f || t->root != f->root)
      return Error("select between pointers into different buffers or shared "
                   "arrays (Vulkan has logical addressing): " +
                   printed(inst));
    Value *zero = b.getInt32(0);
    addresses[select] = {t->root,
                         b.CreateSelect(select->getCondition(),
                                        t->index ? t->index : zero,
                                        f->index ? f->index : zero)};
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
    addresses[gep] = {base->root, combined};
    dead.push_back(gep);
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
      intrinsic->replaceAllUsesWith(ConstantInt::get(
          intrinsic->getType(), localSize[dim->getZExtValue()]));
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
  unsigned pointerOperand = ~0u;
  if (isa<LoadInst>(inst))
    pointerOperand = LoadInst::getPointerOperandIndex();
  else if (isa<StoreInst>(inst))
    pointerOperand = StoreInst::getPointerOperandIndex();
  else if (isa<AtomicRMWInst>(inst))
    pointerOperand = AtomicRMWInst::getPointerOperandIndex();
  else if (isa<AtomicCmpXchgInst>(inst))
    pointerOperand = AtomicCmpXchgInst::getPointerOperandIndex();
  for (Use &use : inst.operands()) {
    std::optional<Address> address = addressOf(use.get());
    if (!address)
      continue;
    if (use.getOperandNo() != pointerOperand)
      return Error("unsupported use of a buffer or shared-memory pointer "
                   "(Vulkan has logical addressing): " +
                   printed(inst));
    const Root &root = roots[address->root];
    if (splitVectorAccess(b, inst, root, address->index))
      return {};
    use.set(getPointer(b, root, address->index));
    retypeAccess(b, inst, root.elementType);
  }
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
  for (auto [phi, index] : phis) {
    for (unsigned i = 0, e = phi->getNumIncomingValues(); i != e; ++i) {
      Value *value = phi->getIncomingValue(i);
      BasicBlock *block = phi->getIncomingBlock(i);
      Value *incoming;
      if (isa<UndefValue>(value))
        incoming = PoisonValue::get(i32);
      else if (auto it = addresses.find(value); it != addresses.end())
        incoming = it->second.index ? it->second.index
                                    : ConstantInt::get(i32, 0);
      else
        return Error("phi mixes buffer or shared-memory pointers with other "
                     "pointers (Vulkan has logical addressing): " +
                     printed(*phi));
      index->addIncoming(incoming, block);
    }
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
  unsigned binding = 0;
  SmallVector<const Leaf *> scalars;
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
    roots.push_back({handle, nullptr, elementType, handleType});
    for (Value *value : leaf.values)
      addresses[value] = {static_cast<unsigned>(roots.size() - 1), nullptr};
    ++binding;
  }
  if (!scalars.empty()) {
    Type *i64 = Type::getInt64Ty(ctx);
    Type *handleType = TargetExtType::get(ctx, "spirv.VulkanBuffer",
                                          {ArrayType::get(i64, 0)},
                                          {kStorageBufferClass, /*writable=*/1});
    Root params{createHandle(b, handleType, binding, "params"), nullptr, i64,
                handleType};
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
      for (Value *use : leaf->values)
        use->replaceAllUsesWith(value);
    }
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
  if (ErrorOrSuccess err = eraseDead())
    return err;

  for (Argument &arg : kernel.args())
    if (!arg.use_empty())
      return Error("kernel argument still in use: " + printed(arg));
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
  return {};
}

} // namespace

ErrorOrSuccess lowerSPIRVKernelsToVulkan(Module &module) {
  ErrorOr<LocalSize> localSize = getLocalSize();
  if (localSize.isError())
    return localSize.takeError();

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
    LocalSize size = *localSize;
    if (Attribute attr = kernel->getFnAttribute("mojo.spirv.numthreads");
        attr.isStringAttribute()) {
      unsigned threads = 0;
      if (attr.getValueAsString().getAsInteger(10, threads) || threads == 0)
        return Error("invalid mojo.spirv.numthreads on @" + kernel->getName());
      size = {threads, 1, 1};
    }
    KernelRewriter rewriter(module, *kernel, name, size);
    if (ErrorOrSuccess err = rewriter.run())
      return err;
  }

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

} // namespace M::KGEN
