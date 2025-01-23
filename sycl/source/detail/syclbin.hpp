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

#include "llvm/Object/SYCLBIN.h"

#include <algorithm>
#include <list>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {

class device;

namespace detail {

inline std::ostream &operator<<(std::ostream &OS,
                                const llvm::object::SYCLBIN::IRType &Type) {
  switch (Type) {
  case llvm::object::SYCLBIN::IRType::SPIRV:
    OS << "SPIRV";
    return OS;
  case llvm::object::SYCLBIN::IRType::PTX:
    OS << "PTX";
    return OS;
  case llvm::object::SYCLBIN::IRType::AMDGCN:
    OS << "AMDGCN";
    return OS;
  default:
    OS << "UNKNOWN";
    return OS;
  }
}

// Helper class for managing both a SYCLBIN and binaries created from it,
// allowing existing infrastructure to better understand the contents of the
// SYCLBINs.
struct SYCLBINBinaries {
  // Delete copy-ctor to keep binaries unique and avoid costly copies of a
  // heavy structure.
  SYCLBINBinaries(const SYCLBINBinaries &) = delete;

  SYCLBINBinaries(std::unique_ptr<llvm::object::SYCLBIN> &&ParsedSYCLBIN);

  SYCLBINBinaries(const char *SYCLBINContent, size_t SYCLBINSize);

  std::vector<const RTDeviceBinaryImage *>
  getBestCompatibleImages(const device &Dev);
  std::vector<const RTDeviceBinaryImage *>
  getBestCompatibleImages(const std::vector<device> &Dev);

  llvm::object::SYCLBIN::BundleState getState() const noexcept {
    return ParsedSYCLBIN->Header.State;
  }

  bool hasKernel(const std::string &Name) {
    return std::any_of(
        ParsedSYCLBIN->AbstractModules.begin(),
        ParsedSYCLBIN->AbstractModules.end(),
        [&](const llvm::object::SYCLBIN::AbstractModule &AM) {
          return std::any_of(
              AM.KernelNames.begin(), AM.KernelNames.end(),
              [&](const llvm::SmallString<0> &KN) { return Name == KN; });
        });
  }

private:
  std::vector<_sycl_offload_entry_struct> &
  convertAbstractModuleEntries(const llvm::object::SYCLBIN::AbstractModule &AM);

  std::vector<_sycl_device_binary_property_set_struct> &
  convertAbstractModuleProperties(
      const llvm::object::SYCLBIN::AbstractModule &AM);

  std::unique_ptr<llvm::object::SYCLBIN> ParsedSYCLBIN;

  // Buffers for holding entries in the binary structs alive. These are
  // multi-layered vectors to avoid relocation of the owned elements.
  std::vector<std::vector<_sycl_offload_entry_struct>> BinaryOffloadEntries;
  std::vector<std::vector<_sycl_device_binary_property_struct>>
      BinaryProperties;
  std::vector<std::vector<_sycl_device_binary_property_set_struct>>
      BinaryPropertySets;
  std::vector<std::vector<std::string>> StringsBuffers;

  // Use std::list to avoid relocations.
  std::list<sycl_device_binary_struct> DeviceBinaries;
  std::list<RTDeviceBinaryImage> JITDeviceBinaryImages;
  std::list<RTDeviceBinaryImage> NativeDeviceBinaryImages;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
