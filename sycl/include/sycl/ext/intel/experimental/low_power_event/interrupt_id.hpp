//==------------------------- SYCL interrupt IDs ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {
class interrupt_id;
} // namespace ext::intel::experimental

namespace detail {
class interrupt_id_impl;

} // namespace detail

namespace ext::intel::experimental {

class interrupt_id {
public:
  interrupt_id() = delete;

  interrupt_id(const interrupt_id &Other) = default;
  interrupt_id(interrupt_id &&Other) = default;

  bool operator==(const interrupt_id &rhs) const { return impl == rhs.impl; }
  bool operator!=(const interrupt_id &rhs) const { return !(*this == rhs); }

private:
  interrupt_id(std::shared_ptr<sycl::detail::interrupt_id_impl> Impl) : impl{Impl} {}

  std::shared_ptr<sycl::detail::interrupt_id_impl> impl;

  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  template <class Obj>
  friend const decltype(Obj::impl) &
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);

  template <backend BackendName, class SyclObjectT>
  friend auto sycl::get_native(const SyclObjectT &Obj)
      -> sycl::backend_return_t<BackendName, SyclObjectT>;
};

} // namespace ext::intel::experimental
} // namespace _V1
} // namespace sycl

namespace std {
template <> struct hash<sycl::ext::intel::experimental::interrupt_id> {
  size_t
  operator()(const sycl::ext::intel::experimental::interrupt_id &IID) const {
    return hash<std::shared_ptr<sycl::detail::interrupt_id_impl>>()(
        sycl::detail::getSyclObjImpl(IID));
  }
};
} // namespace std
