//==--------------------- syclbin.cpp - SYCLBIN parser ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/program_manager/program_manager.hpp>
#include <detail/syclbin.hpp>

#include "llvm/Object/OffloadBinary.h"

namespace sycl {
inline namespace _V1 {
namespace detail {

static std::unique_ptr<llvm::object::SYCLBIN>
ReadSYCLBINOrThrow(const char *SYCLBINContent, size_t SYCLBINSize) {
  std::unique_ptr<llvm::MemoryBuffer> SYCLBINBuff =
      llvm::MemoryBuffer::getMemBuffer(
          llvm::StringRef{SYCLBINContent, SYCLBINSize});
  llvm::MemoryBufferRef SYCLBINBuffRef = *SYCLBINBuff;

  // The actual SYCLBIN format may be contained inside an offload binary. Try
  // to parse that and fall back to parsing SYCLBIN directly if it fails.
  std::unique_ptr<llvm::object::OffloadBinary> ParsedOffloadBinary;
  if (!llvm::object::OffloadBinary::create(SYCLBINBuffRef)
          .moveInto(ParsedOffloadBinary))
    SYCLBINBuffRef = llvm::MemoryBufferRef(ParsedOffloadBinary->getImage(), "");

  std::unique_ptr<llvm::object::SYCLBIN> ParsedSYCLBIN = nullptr;
  if (llvm::Error &&EC =
          llvm::object::SYCLBIN::read(SYCLBINBuffRef).moveInto(ParsedSYCLBIN)) {
    llvm::handleAllErrors(
        std::move(EC), [](const llvm::ErrorInfoBase &EIB) -> llvm::Error {
          throw sycl::exception(make_error_code(errc::invalid),
                                "Failed to read SYCLBIN file: " +
                                    EIB.message());
        });
  }
  return ParsedSYCLBIN;
}

SYCLBINBinaries::SYCLBINBinaries(const char *SYCLBINContent, size_t SYCLBINSize)
    : SYCLBINBinaries(ReadSYCLBINOrThrow(SYCLBINContent, SYCLBINSize)) {}

SYCLBINBinaries::SYCLBINBinaries(std::unique_ptr<llvm::object::SYCLBIN> &&SBIN)
    : ParsedSYCLBIN(std::move(SBIN)) {
  for (llvm::object::SYCLBIN::AbstractModule &AM :
       ParsedSYCLBIN->AbstractModules) {
    // Construct offload entries.
    std::vector<_sycl_offload_entry_struct> &BinaryOffloadEntries =
        convertAbstractModuleEntries(AM);

    // Construct properties from SYCLBIN metadata.
    std::vector<_sycl_device_binary_property_set_struct> &BinPropertySets =
        convertAbstractModuleProperties(AM);

    for (const llvm::object::SYCLBIN::IRModule &IRM : AM.IRModules) {
      DeviceBinaries.emplace_back();
      sycl_device_binary_struct &DeviceBinary = DeviceBinaries.back();
      DeviceBinary.Version = SYCL_DEVICE_BINARY_VERSION;
      DeviceBinary.Kind = 4;
      DeviceBinary.Format = SYCL_DEVICE_BINARY_TYPE_SPIRV; // TODO: Determine.
      DeviceBinary.DeviceTargetSpec =
          __SYCL_DEVICE_BINARY_TARGET_SPIRV64; // TODO: Determine.
      DeviceBinary.CompileOptions = nullptr;
      DeviceBinary.LinkOptions = nullptr;
      DeviceBinary.ManifestStart = nullptr;
      DeviceBinary.ManifestEnd = nullptr;
      DeviceBinary.BinaryStart =
          reinterpret_cast<const unsigned char *>(IRM.RawIRBytes.data());
      DeviceBinary.BinaryEnd = reinterpret_cast<const unsigned char *>(
          IRM.RawIRBytes.data() + IRM.RawIRBytes.size());
      DeviceBinary.EntriesBegin = BinaryOffloadEntries.data();
      DeviceBinary.EntriesEnd =
          BinaryOffloadEntries.data() + BinaryOffloadEntries.size();
      DeviceBinary.PropertySetsBegin = BinPropertySets.data();
      DeviceBinary.PropertySetsEnd =
          BinPropertySets.data() + BinPropertySets.size();
      // Create an image from it.
      JITDeviceBinaryImages.emplace_back(&DeviceBinary);
    }

    for (const llvm::object::SYCLBIN::NativeDeviceCodeImage &NDCI :
         AM.NativeDeviceCodeImages) {
      DeviceBinaries.emplace_back();
      sycl_device_binary_struct &DeviceBinary = DeviceBinaries.back();
      DeviceBinary.Version = SYCL_DEVICE_BINARY_VERSION;
      DeviceBinary.Kind = 4;
      DeviceBinary.Format = SYCL_DEVICE_BINARY_TYPE_NATIVE;
      DeviceBinary.DeviceTargetSpec =
          __SYCL_DEVICE_BINARY_TARGET_UNKNOWN; // TODO: Determine.
      DeviceBinary.CompileOptions = nullptr;
      DeviceBinary.LinkOptions = nullptr;
      DeviceBinary.ManifestStart = nullptr;
      DeviceBinary.ManifestEnd = nullptr;
      DeviceBinary.BinaryStart = reinterpret_cast<const unsigned char *>(
          NDCI.RawDeviceCodeImageBytes.data());
      DeviceBinary.BinaryEnd = reinterpret_cast<const unsigned char *>(
          NDCI.RawDeviceCodeImageBytes.data() +
          NDCI.RawDeviceCodeImageBytes.size());
      DeviceBinary.EntriesBegin = BinaryOffloadEntries.data();
      DeviceBinary.EntriesEnd =
          BinaryOffloadEntries.data() + BinaryOffloadEntries.size();
      DeviceBinary.PropertySetsBegin = BinPropertySets.data();
      DeviceBinary.PropertySetsEnd =
          BinPropertySets.data() + BinPropertySets.size();
      // Create an image from it.
      NativeDeviceBinaryImages.emplace_back(&DeviceBinary);
    }
  }
}

std::vector<_sycl_offload_entry_struct> &
SYCLBINBinaries::convertAbstractModuleEntries(
    const llvm::object::SYCLBIN::AbstractModule &AM) {
  std::vector<_sycl_offload_entry_struct> &BinOffloadEntries =
      BinaryOffloadEntries.emplace_back();
  std::vector<std::string> &StringBuffer = StringsBuffers.emplace_back();
  BinOffloadEntries.reserve(AM.KernelNames.size());
  StringBuffer.reserve(AM.KernelNames.size());

  for (const llvm::SmallString<0> &KernelName : AM.KernelNames){
    _sycl_offload_entry_struct &OffloadEntry = BinOffloadEntries.emplace_back();
    std::string &Str = StringBuffer.emplace_back(KernelName);
    OffloadEntry.name = const_cast<char *>(Str.c_str());
    OffloadEntry.addr = nullptr;
    OffloadEntry.size = 0;
    OffloadEntry.flags = 0;
    OffloadEntry.reserved = 0;
  }

  return BinOffloadEntries;
}

std::vector<_sycl_device_binary_property_set_struct> &
SYCLBINBinaries::convertAbstractModuleProperties(
    const llvm::object::SYCLBIN::AbstractModule &AM) {
  std::vector<_sycl_device_binary_property_set_struct> &BinPropertySets =
      BinaryPropertySets.emplace_back();
  BinPropertySets.reserve(AM.Properties->getPropSets().size());
  for (auto PropSet : *AM.Properties) {
    // Add a new vector to BinaryProperties and add reserve room for all the
    // properties we are converting.
    std::vector<_sycl_device_binary_property_struct> &PropsList =
        BinaryProperties.emplace_back();
    PropsList.reserve(PropSet.second.size());

    // Then convert all properties in the property set.
    for (auto Prop : PropSet.second) {
      _sycl_device_binary_property_struct &BinProp = PropsList.emplace_back();
      BinProp.Name = const_cast<char *>(Prop.first.c_str());
      BinProp.Type = Prop.second.getType();
      BinProp.ValAddr = const_cast<char *>(Prop.second.data());
      BinProp.ValSize = Prop.second.size();
    }

    // Add a new property set to the list.
    _sycl_device_binary_property_set_struct &BinPropSet =
        BinPropertySets.emplace_back();
    BinPropSet.Name = const_cast<char *>(PropSet.first.c_str());
    BinPropSet.PropertiesBegin = PropsList.data();
    BinPropSet.PropertiesEnd = PropsList.data() + PropsList.size();
  }
  return BinPropertySets;
}

std::vector<const RTDeviceBinaryImage *>
SYCLBINBinaries::getBestCompatibleImages(const device &Dev) {
  auto SelectCompatibleImages =
      [&](const std::list<RTDeviceBinaryImage> &Imgs) {
        std::vector<const RTDeviceBinaryImage *> CompatImgs;
        for (const RTDeviceBinaryImage &Img : Imgs)
          if (doesDevSupportDeviceRequirements(Dev, Img) &&
              doesImageTargetMatchDevice(Img, Dev))
            CompatImgs.push_back(&Img);
        return CompatImgs;
      };

  // Try with native images first.
  std::vector<const RTDeviceBinaryImage *> NativeImgs =
      SelectCompatibleImages(NativeDeviceBinaryImages);
  if (!NativeImgs.empty())
    return NativeImgs;

  // If there were no native images, pick JIT images.
  return SelectCompatibleImages(JITDeviceBinaryImages);
}

std::vector<const RTDeviceBinaryImage *>
SYCLBINBinaries::getBestCompatibleImages(const std::vector<device> &Devs) {
  std::set<const RTDeviceBinaryImage *> Images;
  for (const device &Dev : Devs) {
    std::vector<const RTDeviceBinaryImage *> BestImagesForDev =
        getBestCompatibleImages(Dev);
    Images.insert(BestImagesForDev.cbegin(), BestImagesForDev.cend());
  }
  return {Images.cbegin(), Images.cend()};
}

} // namespace detail
} // namespace _V1
} // namespace sycl
