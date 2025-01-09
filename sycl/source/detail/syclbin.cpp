//==--------------------- syclbin.cpp - SYCLBIN parser ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/program_manager/program_manager.hpp>
#include <detail/syclbin.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

SYCLBINBinaries::SYCLBINBinaries(SYCLBIN &&SBIN)
    : ParsedSYCLBIN(std::move(SBIN)) {
  size_t NumJITBinaries = 0, NumNativeBinaries = 0;
  for (const SYCLBIN::AbstractModule &AM : ParsedSYCLBIN.AbstractModules) {
    NumJITBinaries += AM.IRModules.size();
    NumNativeBinaries += AM.NativeDeviceCodeImages.size();
  }
  DeviceBinaries.reserve(NumJITBinaries + NumNativeBinaries);
  JITDeviceBinaryImages.reserve(NumJITBinaries);
  NativeDeviceBinaryImages.reserve(NumNativeBinaries);

  for (SYCLBIN::AbstractModule &AM : ParsedSYCLBIN.AbstractModules) {
    // Construct offload entries.
    std::vector<_sycl_offload_entry_struct> &BinaryOffloadEntries =
        convertAbstractModuleEntries(AM);

    // Construct properties from SYCLBIN metadata.
    std::vector<_sycl_device_binary_property_set_struct> &BinPropertySets =
        convertAbstractModuleProperties(AM);

    for (SYCLBIN::IRModule &IRM : AM.IRModules) {
      sycl_device_binary_struct &DeviceBinary = DeviceBinaries.emplace_back();
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

    for (const SYCLBIN::NativeDeviceCodeImage &NDCI :
         AM.NativeDeviceCodeImages) {
      sycl_device_binary_struct &DeviceBinary = DeviceBinaries.emplace_back();
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
    const SYCLBIN::AbstractModule &AM) {
  std::vector<_sycl_offload_entry_struct> &BinOffloadEntries =
      BinaryOffloadEntries.emplace_back();
  BinOffloadEntries.reserve(AM.KernelNames.size() + AM.ExportedSymbols.size());
  auto InsertEntry = [&](const std::string &EntryName) {
    _sycl_offload_entry_struct &OffloadEntry = BinOffloadEntries.emplace_back();
    OffloadEntry.name = const_cast<char *>(EntryName.c_str());
    OffloadEntry.addr = nullptr;
    OffloadEntry.size = 0;
    OffloadEntry.flags = 0;
    OffloadEntry.reserved = 0;
  };

  for (const std::string &KernelName : AM.KernelNames)
    InsertEntry(KernelName);
  for (const std::string &ExportedSymbol : AM.ExportedSymbols)
    InsertEntry(ExportedSymbol);

  return BinOffloadEntries;
}

std::vector<_sycl_device_binary_property_set_struct> &
SYCLBINBinaries::convertAbstractModuleProperties(SYCLBIN::AbstractModule &AM) {
  std::vector<_sycl_device_binary_property_set_struct> &BinPropertySets =
      BinaryPropertySets.emplace_back();
  BinPropertySets.reserve(AM.Properties.size());
  for (SYCLBIN::PropertySet &PropSet : AM.Properties) {
    // Add a new vector to BinaryProperties and add reserve room for all the
    // properties we are converting.
    std::vector<_sycl_device_binary_property_struct> &PropsList =
        BinaryProperties.emplace_back();
    PropsList.reserve(PropSet.Properties.size());

    // Then convert all properties in the property set.
    for (SYCLBIN::Property &Prop : PropSet.Properties) {
      _sycl_device_binary_property_struct &BinProp = PropsList.emplace_back();
      BinProp.Name = const_cast<char *>(Prop.Name.c_str());
      BinProp.Type = Prop.Type;
      BinProp.ValAddr = Prop.Data.data();
      BinProp.ValSize = Prop.Data.size();
    }

    // Add a new property set to the list.
    _sycl_device_binary_property_set_struct &BinPropSet =
        BinPropertySets.emplace_back();
    BinPropSet.Name = const_cast<char *>(PropSet.Name.c_str());
    BinPropSet.PropertiesBegin = PropsList.data();
    BinPropSet.PropertiesEnd = PropsList.data() + PropsList.size();
  }
  return BinPropertySets;
}

std::vector<const RTDeviceBinaryImage *>
SYCLBINBinaries::getBestCompatibleImages(const device &Dev) {
  auto SelectCompatibleImages =
      [&](const std::vector<RTDeviceBinaryImage> &Imgs) {
        std::vector<const RTDeviceBinaryImage *> CompatImgs;
        for (const RTDeviceBinaryImage &Img : Imgs)
          if (doesDevSupportDeviceRequirements(Dev, Img) &&
              doesImageTargetMatchDevice(Img, Dev))
            CompatImgs.push_back(&Img);
        return CompatImgs;
      };

  // Try with native images first.
  std::vector<const RTDeviceBinaryImage *>
      NativeImgs = SelectCompatibleImages(NativeDeviceBinaryImages);
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
