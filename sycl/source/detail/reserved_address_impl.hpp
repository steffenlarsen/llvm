//==------ reserved_address_impl.hpp - reserved_address implementation -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/context_impl.hpp>
#include <detail/physical_mem_impl.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/reserved_address.hpp>

#include <mutex>
#include <vector>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

class reserved_address_impl {
private:
  static RT::PiVirtualAccessFlags AccessModeToVirtualAccessFlags(
      ext::oneapi::experimental::address_access_mode Mode) {
    switch (Mode) {
    case ext::oneapi::experimental::address_access_mode::read:
      return PI_VIRTUAL_ACCESS_FLAG_READ_ONLY;
    case ext::oneapi::experimental::address_access_mode::read_write:
      return PI_VIRTUAL_ACCESS_FLAG_RW;
    case ext::oneapi::experimental::address_access_mode::none:
      return RT::PiVirtualAccessFlags{0};
    }
  }

  static uintptr_t
  ReserveAddress(uintptr_t RequestedStart, size_t NumBytes,
                 const std::shared_ptr<context_impl> &ContextImpl) {
    const plugin &Plugin = ContextImpl->getPlugin();
    void *OutPtr = nullptr;
    Plugin.call<PiApiKind::piextVirtualMemReserve>(
        ContextImpl->getHandleRef(), reinterpret_cast<void *>(RequestedStart),
        NumBytes, &OutPtr);
    return reinterpret_cast<uintptr_t>(OutPtr);
  }

public:
  reserved_address_impl(uintptr_t RequestedStart, size_t NumBytes,
                        const context &SyclContext)
      : MContext(getSyclObjImpl(SyclContext)), MNumBytes{NumBytes},
        MStart{ReserveAddress(RequestedStart, MNumBytes, MContext)} {}

  ~reserved_address_impl() {
    // First unmap all ranges. We let the vector destructor clear it out after.
    UnmapAll();

    // Then free the memory.
    const plugin &Plugin = MContext->getPlugin();
    Plugin.call<PiApiKind::piextVirtualMemFree>(
        MContext->getHandleRef(), reinterpret_cast<void *>(MStart), MNumBytes);
  }

  void *map(size_t RangeOffset, size_t RangeSize,
            const ext::oneapi::experimental::physical_mem &PhysicalMem,
            size_t PhysicalOffset,
            ext::oneapi::experimental::address_access_mode Mode) {
    RT::PiVirtualAccessFlags AccessFlags = AccessModeToVirtualAccessFlags(Mode);
    void *RangeStart = reinterpret_cast<void *>(MStart + RangeOffset);
    std::shared_ptr<physical_mem_impl> PhysicalMemImpl =
        getSyclObjImpl(PhysicalMem);
    const plugin &Plugin = MContext->getPlugin();
    Plugin.call<PiApiKind::piextVirtualMemMap>(
        MContext->getHandleRef(), RangeStart, RangeSize,
        PhysicalMemImpl->getHandleRef(), PhysicalOffset, AccessFlags);

    const std::lock_guard<std::mutex> Lock{MMappedRangesMutex};
    // Add the mapped range to the tracked collection.
    MMappedRanges.emplace_back(
        MappedRange{RangeStart, RangeSize, std::move(PhysicalMemImpl)});
    return RangeStart;
  }

  void unmap() {
    const std::lock_guard<std::mutex> Lock{MMappedRangesMutex};
    UnmapAll();
    MMappedRanges.clear();
  }

  void unmap(size_t RangeOffset) {
    const std::lock_guard<std::mutex> Lock{MMappedRangesMutex};

    // Find the corresponding mapped range.
    void *RangeStart = reinterpret_cast<void *>(MStart + RangeOffset);
    auto FoundMappedRange =
        std::find_if(MMappedRanges.begin(), MMappedRanges.end(),
                     [&](const MappedRange &AddrRange) {
                       return AddrRange.RangeStart == RangeStart;
                     });

    // If it doesn't exist, throw an exception.
    if (FoundMappedRange == MMappedRanges.end())
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::invalid),
          "No range at the specified offset is currently mapped.");

    // If it exists, unmap it and remove it from the range.
    const plugin &Plugin = MContext->getPlugin();
    Plugin.call<PiApiKind::piextVirtualMemUnmap>(MContext->getHandleRef(),
                                                 FoundMappedRange->RangeStart,
                                                 FoundMappedRange->RangeSize);
    MMappedRanges.erase(FoundMappedRange, FoundMappedRange);
  }

  void unmap(const ext::oneapi::experimental::physical_mem &PhysicalMem) {
    std::shared_ptr<physical_mem_impl> PhysicalMemImpl =
        getSyclObjImpl(PhysicalMem);

    const std::lock_guard<std::mutex> Lock{MMappedRangesMutex};

    // Find the corresponding mapped range.
    auto FoundMappedRange =
        std::find_if(MMappedRanges.begin(), MMappedRanges.end(),
                     [&](const MappedRange &AddrRange) {
                       return AddrRange.PhysicalMemImpl == PhysicalMemImpl;
                     });

    // If it doesn't exist, throw an exception.
    if (FoundMappedRange == MMappedRanges.end())
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "No sub-range of is currently mapped to the "
                            "specified physical memory.");

    // If it exists, unmap it and remove it from the range.
    const plugin &Plugin = MContext->getPlugin();
    Plugin.call<PiApiKind::piextVirtualMemUnmap>(MContext->getHandleRef(),
                                                 FoundMappedRange->RangeStart,
                                                 FoundMappedRange->RangeSize);
    MMappedRanges.erase(FoundMappedRange, FoundMappedRange);
  }

  void set_access_mode(ext::oneapi::experimental::address_access_mode Access) {
    RT::PiVirtualAccessFlags AccessFlags =
        AccessModeToVirtualAccessFlags(Access);
    const plugin &Plugin = MContext->getPlugin();

    const std::lock_guard<std::mutex> Lock{MMappedRangesMutex};
    for (const MappedRange &AddrRange : MMappedRanges) {
      Plugin.call<PiApiKind::piextVirtualMemSetAccess>(
          MContext->getHandleRef(), AddrRange.RangeStart, AddrRange.RangeSize,
          AccessFlags);
    }
  }

  ext::oneapi::experimental::address_access_mode get_access_mode() const {
    const std::lock_guard<std::mutex> Lock{MMappedRangesMutex};

    if (MMappedRanges.empty())
      throw ext::oneapi::experimental::address_access_mode::none;

    // The access mode of all mapped ranges must be the same. We just query one.
    const MappedRange &AddrRange = MMappedRanges[0];

    const plugin &Plugin = MContext->getPlugin();
#ifndef NDEBUG
    size_t InfoOutputSize;
    Plugin.call<PiApiKind::piextVirtualMemAccessGetInfo>(
        MContext->getHandleRef(), AddrRange.RangeStart, AddrRange.RangeSize,
        PI_EXT_ONEAPI_VIRTUAL_MEM_ACCESS_INFO_ACCESS_MODE, 0, nullptr,
        &InfoOutputSize);
    assert(InfoOutputSize == sizeof(RT::PiVirtualAccessFlags) &&
           "Unexpected output size of access mode info query.");
#endif // NDEBUG
    RT::PiVirtualAccessFlags AccessFlags;
    Plugin.call<PiApiKind::piextVirtualMemAccessGetInfo>(
        MContext->getHandleRef(), AddrRange.RangeStart, AddrRange.RangeSize,
        PI_EXT_ONEAPI_VIRTUAL_MEM_ACCESS_INFO_ACCESS_MODE,
        sizeof(RT::PiVirtualAccessFlags), &AccessFlags, nullptr);

    if (AccessFlags & PI_VIRTUAL_ACCESS_FLAG_RW)
      return ext::oneapi::experimental::address_access_mode::read_write;
    if (AccessFlags & PI_VIRTUAL_ACCESS_FLAG_READ_ONLY)
      return ext::oneapi::experimental::address_access_mode::read;
    return ext::oneapi::experimental::address_access_mode::none;
  }

  uintptr_t get_start() const noexcept { return MStart; }
  context get_context() const {
    return createSyclObjFromImpl<context>(MContext);
  }
  size_t size() const noexcept { return MNumBytes; }

private:
  const std::shared_ptr<context_impl> MContext;
  const size_t MNumBytes;
  const uintptr_t MStart;

  struct MappedRange {
    const void *RangeStart;
    size_t RangeSize;
    std::shared_ptr<physical_mem_impl> PhysicalMemImpl;
  };

  std::vector<MappedRange> MMappedRanges;
  mutable std::mutex MMappedRangesMutex;

  // Unmaps all mapped ranges. MMappedRangesMutex must be held when this is
  // called.
  void UnmapAll() {
    const plugin &Plugin = MContext->getPlugin();
    for (const MappedRange &AddrRange : MMappedRanges) {
      Plugin.call<PiApiKind::piextVirtualMemUnmap>(
          MContext->getHandleRef(), AddrRange.RangeStart, AddrRange.RangeSize);
    }
  }
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
