//==----- properties.hpp - SYCL properties associated with submissions -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::experimental {

struct eventless_key {
  using value_t = property_value<eventless_key>;
};

inline constexpr eventless_key::value_t eventless;

template <> struct is_property_key<eventless_key> : std::true_type {};

namespace detail {

template <> struct PropertyToKind<eventless_key> {
  static constexpr PropKind Kind = PropKind::Eventless;
};

template <> struct IsCompileTimeProperty<eventless_key> : std::true_type {};

} // namespace detail
} // namespace ext::oneapi::experimental

class event;

namespace detail {

template <typename PropertiesT, typename Cond = void> struct HasEventlessProp {
  static constexpr bool value = false;
};

template <typename PropertiesT>
struct HasEventlessProp<
    PropertiesT, std::enable_if_t<ext::oneapi::experimental::is_property_list_v<
                     PropertiesT>>> {
  static constexpr bool value =
      PropertiesT::template has_property<ext::oneapi::experimental::eventless_key>();
};

template <typename PropertiesT>
using SubmitReturnType =
    std::conditional_t<HasEventlessProp<PropertiesT>::value, void, event>;

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
