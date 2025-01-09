//===- SYCLBIN.h - SYCLBIN binary format support ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_SYCLBIN_H
#define LLVM_OBJECT_SYCLBIN_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/SYCLLowerIR/ModuleSplitter.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>

namespace llvm {

namespace object {

class SYCLBIN : public Binary {
public:
  enum class ModuleState : uint8_t { Input, Object, Executable };

  struct ModuleDesc {
    ModuleState State;
    std::string ArchString;
    std::vector<module_split::SplitModule> SplitModules;
  };

  /// The current version of the binary used for backwards compatibility.
  static constexpr uint32_t Version = 1;

  /// Magic number used to identify SYCLBIN files.
  static constexpr uint8_t MagicNumber[4] = {0x53, 0x59, 0x42, 0x49};

  /// Serialize the contents of \p ModuleDescs to a binary buffer to be read
  /// later.
  static Expected<SmallString<0>> write(const SmallVector<ModuleDesc> &);

  static uint64_t getAlignment() { return 8; }

  static bool classof(const Binary *V) { return V->isSYCLBINFile(); }

private:
  struct Header {
    uint8_t Magic[4] = {MagicNumber[0], MagicNumber[1], MagicNumber[2],
                        MagicNumber[3]}; // 0x53594249 magic bytes.
    uint32_t Version = SYCLBIN::Version; // Version identifier.
    uint8_t State;                       // The state of the bundle.
  };

  struct StringEntry {
    uint16_t Size;
    char *Data;
  };

  struct AbstractModuleMetadata {
    SmallVector<StringEntry *, 4> KernelNames;
    SmallVector<StringEntry *, 4> ImportedSymbols;
    SmallVector<StringEntry *, 4> ExportedSymbols;
    uint32_t PropertySetSize;
    char *PropertySetData;
  };

  enum class IRType : uint8_t { SPIRV = 0, PTX = 1, AMDGCN = 2 };

  struct IRModule {
    IRType Type;
    uint32_t BinarySize;
    char *BinaryData;
  };

  struct NativeDeviceCodeImage {
    StringEntry Architecture;
    uint32_t BinarySize;
    char *BinaryData;
  };

  struct AbstractModule {
    AbstractModuleMetadata *Metadata;
    SmallVector<IRModule *, 4> IRModules;
    SmallVector<NativeDeviceCodeImage *, 4> NativeDeviceCodeImages;
  };

  SYCLBIN(MemoryBufferRef Source, const Header *TheHeader,
          const uint32_t NumAbstractModules,
          const AbstractModule *AbstractModuleList)
      : Binary(Binary::ID_SYCLBIN, Source), Buffer(Source.getBufferStart()),
        TheHeader(TheHeader), NumAbstractModules(NumAbstractModules),
        AbstractModuleList(AbstractModuleList) {}

  SYCLBIN(const OffloadBinary &Other) = delete;

  /// Raw pointer to the MemoryBufferRef for convenience.
  const char *Buffer;
  /// Location of the header within the binary.
  const Header *TheHeader;
  /// Number of abstract modules within the binary.
  const uint32_t NumAbstractModules;
  /// Location of the abstract modules within the binary.
  const AbstractModule *AbstractModuleList;
};

} // namespace object

} // namespace llvm

#endif
