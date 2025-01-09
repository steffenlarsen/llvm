//==--------------------- syclbin.hpp - SYCLBIN parser ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "detail/compiler.hpp"
#include "detail/device_binary_image.hpp"
#include "sycl/exception.hpp"

#include <algorithm>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {

class device;

namespace detail {

namespace {

class ConsumerParser {
public:
  ConsumerParser(const char *Data, size_t Size)
      : Data{Data}, RemainingSize{Size} {}

  // This ctor "steals" the next Size bytes from another ConsumerParser object.
  ConsumerParser(ConsumerParser &Other, size_t Size)
      : ConsumerParser(Other.GetCurrentPointer(), Size) {
    Other.Skip(Size);
  }

  void ThrowIfSizeUnavailable(size_t Size) const {
    if (RemainingSize < Size)
      throw sycl::exception(make_error_code(errc::invalid),
                            "File content size mismatch.");
  }

  void Consume(void *Dest, size_t Size) {
    ThrowIfSizeUnavailable(Size);
    std::memcpy(Dest, Data, Size);
    Move(Size);
  }

  template <typename T> T ConsumeScalar() {
    T ReadVal{};
    Consume(&ReadVal, sizeof(T));
    return ReadVal;
  }

  // A common case is where we need to read a size and make sure that size is
  // available after that piece of memory. We call this a "size promise".
  template <typename SizeT> SizeT ConsumeReadSizePromise() {
    static_assert(std::is_integral_v<SizeT>);
    SizeT ReadSize = ConsumeScalar<SizeT>();
    ThrowIfSizeUnavailable(ReadSize);
    return ReadSize;
  }

  template <typename ReadSizePromiseT> std::string_view ConsumeStringView() {
    ReadSizePromiseT StringSize = ConsumeReadSizePromise<ReadSizePromiseT>();
    std::string_view Result{GetCurrentPointer(), size_t{StringSize}};
    Move(StringSize);
    return Result;
  }

  template <typename ReadSizePromiseT> std::string ConsumeString() {
    return static_cast<std::string>(ConsumeStringView<ReadSizePromiseT>());
  }

  std::vector<std::string> ConsumeStringList() {
    uint64_t ByteSize = ConsumeReadSizePromise<uint64_t>();
    ThrowIfSizeUnavailable(ByteSize);

    ConsumerParser ListConsumer(*this, ByteSize);

    uint32_t NumStrings = ListConsumer.ConsumeScalar<uint32_t>();
    std::vector<std::string> Result;
    Result.reserve(NumStrings);
    for (size_t I = 0; I < NumStrings; ++I)
      Result.emplace_back(ListConsumer.ConsumeString<uint32_t>());

    return Result;
  }

  void Skip(size_t Size) {
    ThrowIfSizeUnavailable(Size);
    Move(Size);
  }

  size_t GetRemainingSize() const noexcept { return RemainingSize; }

  bool Empty() const noexcept { return GetRemainingSize() == 0; }

  const char *GetCurrentPointer() const noexcept { return Data; }

private:
  void Move(size_t Size) noexcept {
    Data += Size;
    RemainingSize -= Size;
  }

  const char *Data;
  size_t RemainingSize;
};

} // namespace

struct SYCLBIN {

  enum class IRType : uint8_t { SPIRV = 0, PTX = 1, AMDGCN = 2 };

  struct IRModule {
    IRType Type;
    std::vector<char> RawIRBytes;
  };
  struct NativeDeviceCodeImage {
    std::string ArchString;
    std::vector<char> RawDeviceCodeImageBytes;
  };

  struct Property {
    std::string Name;
    uint32_t Type;
    std::vector<char> Data;
  };
  struct PropertySet {
    std::string Name;
    std::vector<Property> Properties;
  };

  struct AbstractModule {
    std::vector<std::string> KernelNames;
    std::vector<std::string> ImportedSymbols;
    std::vector<std::string> ExportedSymbols;
    std::vector<PropertySet> Properties;

    std::vector<IRModule> IRModules;
    std::vector<NativeDeviceCodeImage> NativeDeviceCodeImages;
  };

  struct {
    uint8_t Magic[4];
    uint32_t Version;
    uint8_t State;
  } Header;

  std::vector<AbstractModule> AbstractModules;
};

inline std::ostream &operator<<(std::ostream &OS, const SYCLBIN::IRType &Type) {
  switch (Type) {
  case SYCLBIN::IRType::SPIRV:
    OS << "SPIRV";
    return OS;
  case SYCLBIN::IRType::PTX:
    OS << "PTX";
    return OS;
  case SYCLBIN::IRType::AMDGCN:
    OS << "AMDGCN";
    return OS;
  default:
    OS << "UNKNOWN";
    return OS;
  }
}

inline std::vector<SYCLBIN::PropertySet>
ParseProperties(ConsumerParser &Consumer) {
  std::string_view PropertiesStringView =
      Consumer.ConsumeStringView<uint32_t>();

  // Get views of all lines.
  std::vector<std::string_view> Lines;
  Lines.reserve(std::count(PropertiesStringView.cbegin(),
                           PropertiesStringView.cend(), '\n') +
                1);
  std::vector<size_t> PropertySetLineStarts;
  for (size_t Pos = 0; Pos != std::string_view::npos;) {
    size_t NextNewline = PropertiesStringView.find('\n', Pos);
    size_t LineSize =
        NextNewline != std::string_view::npos ? NextNewline - Pos : NextNewline;
    std::string_view Line = PropertiesStringView.substr(Pos, LineSize);
    if (!Line.empty()) {
      if (Line.front() == '[')
        PropertySetLineStarts.push_back(Lines.size());
      Lines.emplace_back(std::move(Line));
    }
    Pos = NextNewline + (NextNewline != std::string_view::npos);
  }

  std::vector<SYCLBIN::PropertySet> Result;
  Result.reserve(PropertySetLineStarts.size());
  for (size_t I = 0; I < PropertySetLineStarts.size(); ++I) {
    size_t CurrentStart = PropertySetLineStarts[I];
    size_t NumLines =
        (I + 1 < PropertySetLineStarts.size() ? PropertySetLineStarts[I + 1]
                                              : Lines.size()) -
        CurrentStart;

    SYCLBIN::PropertySet &PropSet = Result.emplace_back();

    // First line is the property set name.
    std::string_view PropertySetName = Lines[CurrentStart];
    if (PropertySetName.front() != '[' || PropertySetName.back() != ']')
      throw sycl::exception(make_error_code(errc::invalid),
                            "Ill-formed property set name");
    PropSet.Name = PropertySetName.substr(1, PropertySetName.size() - 2);

    // The rest are the properties.
    PropSet.Properties.resize(NumLines - 1);
    for (size_t J = 0; J < NumLines - 1; ++J) {
      SYCLBIN::Property &Prop = PropSet.Properties[J];
      std::string_view PropertyLine = Lines[CurrentStart + 1 + J];

      // First is the property name.
      size_t PropertyNameEnd = PropertyLine.find('=');
      if (PropertyNameEnd == std::string_view::npos)
        throw sycl::exception(make_error_code(errc::invalid),
                              "Ill-formed property");
      Prop.Name = PropertyLine.substr(0, PropertyNameEnd);

      // Then is the property type.
      size_t PropertyTypeEnd = PropertyLine.find('|');
      if (PropertyNameEnd == std::string_view::npos)
        throw sycl::exception(make_error_code(errc::invalid),
                              "Ill-formed property");
      Prop.Type = std::stoi(static_cast<std::string>(
          PropertyLine.substr(PropertyNameEnd + 1, PropertyTypeEnd)));

      if (Prop.Type == 2) {
        // Byte arrays need base64 decoding.
        // TODO: Decode base64.
      } else {
        // Rest is the data.
        std::string_view PropertyData =
            PropertyLine.substr(PropertyTypeEnd + 1);
        Prop.Data.resize(PropertyData.size());
        PropertyData.copy(Prop.Data.data(), Prop.Data.size());
      }
    }
  }
  return Result;
}

inline SYCLBIN ParseSYCLBIN(const char *SYCLBINContent, size_t SYCLBINSize) {
  SYCLBIN Result{};
  ConsumerParser DataConsumer{SYCLBINContent, SYCLBINSize};

  // Read header.
  DataConsumer.Consume(Result.Header.Magic, 4 * sizeof(uint8_t));
  constexpr uint8_t SYCLBINMagic[4] = {0x53, 0x59, 0x42, 0x49};
  if (std::memcmp(Result.Header.Magic, SYCLBINMagic, 4) != 0)
    throw sycl::exception(make_error_code(errc::invalid),
                          "Incorrect SYCLBIN magic number");

  Result.Header.Version = DataConsumer.ConsumeScalar<uint32_t>();

  if (Result.Header.Version != 1)
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Unsupported SYCLBIN version " +
                              std::to_string(Result.Header.Version));

  Result.Header.State = DataConsumer.ConsumeScalar<uint8_t>();

  uint64_t BodySize = DataConsumer.ConsumeReadSizePromise<uint64_t>();

  ConsumerParser BodyConsumer{DataConsumer, BodySize};
  while (!BodyConsumer.Empty()) {
    SYCLBIN::AbstractModule &AbstractModule =
        Result.AbstractModules.emplace_back();

    // Abstract module metadata.
    BodyConsumer.ConsumeReadSizePromise<uint64_t>();

    AbstractModule.KernelNames = BodyConsumer.ConsumeStringList();
    AbstractModule.ImportedSymbols = BodyConsumer.ConsumeStringList();
    AbstractModule.ExportedSymbols = BodyConsumer.ConsumeStringList();

    AbstractModule.Properties = ParseProperties(BodyConsumer);

    // IR modules.
    uint64_t IRModuleListSize = BodyConsumer.ConsumeReadSizePromise<uint64_t>();
    ConsumerParser IRModuleListConsumer{BodyConsumer, IRModuleListSize};
    while (!IRModuleListConsumer.Empty()) {
      SYCLBIN::IRModule &IRModule = AbstractModule.IRModules.emplace_back();
      IRModule.Type = IRModuleListConsumer.ConsumeScalar<SYCLBIN::IRType>();
      uint64_t BinarySize =
          IRModuleListConsumer.ConsumeReadSizePromise<uint64_t>();
      IRModule.RawIRBytes.resize(BinarySize);
      IRModuleListConsumer.Consume(IRModule.RawIRBytes.data(), BinarySize);
    }

    // Native device code images.
    uint64_t NDCIListSize = BodyConsumer.ConsumeReadSizePromise<uint64_t>();
    ConsumerParser NDCIListConsumer{BodyConsumer, NDCIListSize};
    while (!NDCIListConsumer.Empty()) {
      SYCLBIN::NativeDeviceCodeImage &NDCI =
          AbstractModule.NativeDeviceCodeImages.emplace_back();
      NDCI.ArchString = NDCIListConsumer.ConsumeString<uint32_t>();
      uint64_t BinarySize = NDCIListConsumer.ConsumeReadSizePromise<uint64_t>();
      NDCI.RawDeviceCodeImageBytes.resize(BinarySize);
      NDCIListConsumer.Consume(NDCI.RawDeviceCodeImageBytes.data(), BinarySize);
    }
  }

  return Result;
}

// Helper class for managing both a SYCLBIN and binaries created from it,
// allowing existing infrastructure to better understand the contents of the
// SYCLBINs.
struct SYCLBINBinaries {
  // Delete copy-ctor to keep binaries unique and avoid costly copies of a
  // heavy structure.
  SYCLBINBinaries(const SYCLBINBinaries &) = delete;

  SYCLBINBinaries(SYCLBIN &&ParsedSYCLBIN);

  SYCLBINBinaries(const char *SYCLBINContent, size_t SYCLBINSize)
      : SYCLBINBinaries(ParseSYCLBIN(SYCLBINContent, SYCLBINSize)) {}

  std::vector<const RTDeviceBinaryImage *>
  getBestCompatibleImages(const device &Dev);
  std::vector<const RTDeviceBinaryImage *>
  getBestCompatibleImages(const std::vector<device> &Dev);

  uint8_t getState() const noexcept {
    return ParsedSYCLBIN.Header.State;
  }

private:
  std::vector<_sycl_offload_entry_struct> &
  convertAbstractModuleEntries(const SYCLBIN::AbstractModule &AM);

  std::vector<_sycl_device_binary_property_set_struct> &
  convertAbstractModuleProperties(SYCLBIN::AbstractModule &AM);

  SYCLBIN ParsedSYCLBIN;

  // Buffers for holding entries in the binary structs alive.
  std::vector<std::vector<_sycl_offload_entry_struct>> BinaryOffloadEntries;
  std::vector<std::vector<_sycl_device_binary_property_struct>>
      BinaryProperties;
  std::vector<std::vector<_sycl_device_binary_property_set_struct>>
      BinaryPropertySets;

  std::vector<sycl_device_binary_struct> DeviceBinaries;
  std::vector<RTDeviceBinaryImage> JITDeviceBinaryImages;
  std::vector<RTDeviceBinaryImage> NativeDeviceBinaryImages;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
