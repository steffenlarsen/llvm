//===- SYCLBIN.cpp - SYCLBIN binary format support --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/SYCLBIN.h"

#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

namespace {

template <typename T>
void BinaryWriteInteger(raw_ostream &OS, T Val) {
  static_assert(std::is_integral_v<T>);
  OS << StringRef(reinterpret_cast<const char *>(&Val), sizeof(T));
}

template <typename SizeType, typename BlockFunc>
void SizedBlockWrite(raw_ostream &OS, const BlockFunc &F) {
  SmallString<0> BlockData = F();
  BinaryWriteInteger(OS, static_cast<SizeType>(BlockData.size()));
  OS << BlockData;
}

} // namespace

Expected<SmallString<0>>
SYCLBIN::write(const SmallVector<SYCLBIN::ModuleDesc> &ModuleDescs) {
  // TODO: Merge by properties and kernel names, so overlap can live in the same
  //       abstract module.

  SmallString<0> Data;
  raw_svector_ostream OS(Data);
  OS << StringRef(reinterpret_cast<const char *>(&MagicNumber),
                  sizeof(MagicNumber));
  BinaryWriteInteger<uint32_t>(OS, Version);
  BinaryWriteInteger<uint8_t>(OS, 0); // TODO: Deduce this from arguments.

  {
    SmallString<0> BodyData;
    raw_svector_ostream BodyOS(BodyData);
    for (const ModuleDesc &Desc : ModuleDescs) {
      for (const module_split::SplitModule &SM : Desc.SplitModules) {
        // Write the abstract module metadata block.
        SmallString<0> AbstractModuleData;
        raw_svector_ostream AbstractModuleOS(AbstractModuleData);

        // Write kernel name string list.
        SizedBlockWrite<uint64_t>(AbstractModuleOS, [&]() {
          SmallString<0> KernelNamesData;
          uint32_t StringCount = 0;
          {
            raw_svector_ostream KernelNamesOS(KernelNamesData);

            size_t CurrentSymbolPos = 0;
            size_t NextSeperator = 0;
            do {
              NextSeperator = SM.Symbols.find('\n', CurrentSymbolPos);
              size_t CurrentSymbolEnd =
                  (NextSeperator != std::string::npos ? NextSeperator
                                                      : SM.Symbols.size());
              size_t CurrentSymbolSize = CurrentSymbolEnd - CurrentSymbolPos;
              if (CurrentSymbolSize) {
                BinaryWriteInteger(KernelNamesOS,
                                   static_cast<uint32_t>(CurrentSymbolSize));
                KernelNamesOS << StringRef(
                    SM.Symbols.c_str() + CurrentSymbolPos, CurrentSymbolSize);
                ++StringCount;
              }
              CurrentSymbolPos = CurrentSymbolEnd + 1;
            } while (NextSeperator != std::string::npos &&
                     CurrentSymbolPos < SM.Symbols.size());
          }

          SmallString<0> FullKernelNamesData;
          FullKernelNamesData.reserve(sizeof(uint32_t) +
                                      KernelNamesData.size());
          raw_svector_ostream FullKernelNamesOS(FullKernelNamesData);
          BinaryWriteInteger<uint32_t>(FullKernelNamesOS, StringCount);
          FullKernelNamesOS << KernelNamesData;
          return FullKernelNamesData;
        });

        // Write imported symbols string list.
        // TODO: Currently empty, so the list byte size is the size of the
        //       string count.
        BinaryWriteInteger<uint64_t>(AbstractModuleOS, 4);
        BinaryWriteInteger<uint32_t>(AbstractModuleOS, 0);

        // Write exported symbols string list.
        // TODO: Currently empty, so the list byte size is the size of the
        //       string count.
        BinaryWriteInteger<uint64_t>(AbstractModuleOS, 4);
        BinaryWriteInteger<uint32_t>(AbstractModuleOS, 0);

        SizedBlockWrite<uint32_t>(AbstractModuleOS, [&]() {
          SmallString<0> PropertiesData;
          raw_svector_ostream PropsOS(PropertiesData);
          SM.Properties.write(PropsOS);
          return PropertiesData;
        });

        // Read the module data. This is needed no matter what kind of module it
        // is.
        auto BinaryDataOrError =
            llvm::MemoryBuffer::getFileOrSTDIN(SM.ModuleFilePath);
        if (std::error_code EC = BinaryDataOrError.getError())
          return createFileError(SM.ModuleFilePath, EC);
        SmallString<0> RawModuleData =
            StringRef((*BinaryDataOrError)->getBufferStart(),
                      (*BinaryDataOrError)->getBufferSize());

        // IR Modules
        SizedBlockWrite<uint64_t>(AbstractModuleOS, [&]() {
          SmallString<0> IRModuleData;
          // If no arch string is present, the module must be IR.
          if (!Desc.ArchString.empty())
            return IRModuleData;
          IRModuleData.reserve(sizeof(IRType) + sizeof(uint64_t) +
                               RawModuleData.size());
          raw_svector_ostream IRModuleOS(IRModuleData);
          BinaryWriteInteger<uint8_t>(IRModuleOS, 0); // TODO: Determine.
          BinaryWriteInteger<uint64_t>(IRModuleOS, RawModuleData.size());
          IRModuleOS << RawModuleData;
          return IRModuleData;
        });

        // Native device code images
        SizedBlockWrite<uint64_t>(AbstractModuleOS, [&]() {
          SmallString<0> NDCIData;
          if (Desc.ArchString.empty())
            return NDCIData;
          NDCIData.reserve(sizeof(uint32_t) + Desc.ArchString.size() +
                           sizeof(uint64_t) + RawModuleData.size());
          raw_svector_ostream NDCIOS(NDCIData);
          BinaryWriteInteger<uint32_t>(NDCIOS, Desc.ArchString.size());
          NDCIOS << Desc.ArchString;
          BinaryWriteInteger<uint64_t>(NDCIOS, RawModuleData.size());
          NDCIOS << RawModuleData;
          return NDCIData;
        });

        // Write abstract modules to body.
        BinaryWriteInteger<uint64_t>(BodyOS, AbstractModuleData.size());
        BodyOS << AbstractModuleData;
      }
    }
    BinaryWriteInteger<uint64_t>(OS, BodyData.size());
    OS << BodyData;
  }

  // Add final padding to required alignment.
  size_t AlignedSize = alignTo(OS.tell(), getAlignment());
  OS.write_zeros(AlignedSize - OS.tell());
  assert(AlignedSize == OS.tell() && "Size mismatch");

  return Data;
}
