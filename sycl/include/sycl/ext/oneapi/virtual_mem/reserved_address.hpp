//==-- reserved_address.hpp - sycl_ext_oneapi_virtual_mem reserved_address -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/owner_less_base.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace detail {
class reserved_address_impl;
} // namespace detail

namespace ext::oneapi::experimental {

enum class address_access_mode : uint32_t { none, read, read_write };

class __SYCL_EXPORT reserved_address
    : public sycl::detail::OwnerLessBase<reserved_address> {
public:
  reserved_address(uintptr_t RequestedStart, size_t NumBytes,
                   const context &SyclContext);

  reserved_address(const reserved_address &rhs) = default;
  reserved_address(reserved_address &&rhs) = default;

  reserved_address &operator=(const reserved_address &Rhs) = default;
  reserved_address &operator=(reserved_address &&Rhs) = default;

  ~reserved_address() = default;

  bool operator==(const reserved_address &Rhs) const {
    return impl == Rhs.impl;
  }
  bool operator!=(const reserved_address &Rhs) const { return !(*this == Rhs); }

  void *map(const physical_mem &PhysicalMem, size_t Offset = 0) {
    return map(0, size(), PhysicalMem, Offset, address_access_mode::none);
  }
  void *map(size_t RangeOffset, size_t RangeSize,
            const physical_mem &PhysicalMem, size_t Offset = 0) {
    return map(RangeOffset, RangeSize, PhysicalMem, Offset,
               address_access_mode::none);
  }
  void *map(size_t RangeOffset, size_t RangeSize,
            const physical_mem &PhysicalMem, size_t Offset,
            address_access_mode Mode = address_access_mode::none);

  void unmap();
  void unmap(size_t RangeOffset);
  void unmap(const physical_mem &PhysicalMem);

  void set_access_mode(address_access_mode Access);
  address_access_mode get_access_mode() const;

  uintptr_t get_start() const noexcept;
  context get_context() const;

  size_t size() const noexcept;

private:
  std::shared_ptr<sycl::detail::reserved_address_impl> impl;

  template <class Obj>
  friend decltype(Obj::impl)
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
};

} // namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

namespace std {
template <> struct hash<sycl::ext::oneapi::experimental::reserved_address> {
  size_t operator()(const sycl::ext::oneapi::experimental::reserved_address
                        &ReservedAddress) const {
    return hash<std::shared_ptr<sycl::detail::reserved_address_impl>>()(
        sycl::detail::getSyclObjImpl(ReservedAddress));
  }
};
} // namespace std
