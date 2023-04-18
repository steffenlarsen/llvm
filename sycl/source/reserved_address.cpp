//==-- reserved_address.cpp - sycl_ext_oneapi_virtual_mem reserved_address -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/reserved_address_impl.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::experimental {

reserved_address::reserved_address(uintptr_t RequestedStart, size_t NumBytes,
                                   const context &SyclContext) {
  std::vector<device> Devs = SyclContext.get_devices();
  if (std::any_of(Devs.cbegin(), Devs.cend(), [](const device &Dev) {
        return !Dev.has(aspect::ext_oneapi_virtual_mem);
      }))
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "One or more devices in the supplied context does not support "
        "aspect::ext_oneapi_virtual_mem.");

  impl = std::make_shared<sycl::detail::reserved_address_impl>(
      RequestedStart, NumBytes, SyclContext);
}

void *reserved_address::map(size_t RangeOffset, size_t RangeSize,
                            const physical_mem &PhysicalMem,
                            size_t PhysicalOffset, address_access_mode Mode) {
  if (RangeOffset + RangeSize > MNumBytes)
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "Specified range is outside the reserved address range.");

  if (PhysicalOffset + RangeSize > PhysicalMem.size())
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Specified range is outside the physical memory.");

  return impl->map(RangeOffset, RangeSize, PhysicalMem, Offset, Mode);
}

void reserved_address::unmap() { impl->unmap(); }
void reserved_address::unmap(size_t RangeOffset) {
  impl->unmap(RangeOffset, RangeSize);
}
void reserved_address::unmap(const physical_mem &PhysicalMem) {
  impl->unmap(RangeOffset, RangeSize);
}

void reserved_address::set_access_mode(address_access_mode Access) {
  impl->set_access_mode(Access);
}
address_access_mode reserved_address::get_access_mode() const {
  return impl->get_access_mode(Access);
}

uintptr_t reserved_address::get_start() const noexcept {
  return impl->get_start();
}
context reserved_address::get_context() const { return impl->get_context(); }
size_t reserved_address::size() const noexcept { return impl->size(); }

} // namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
