//==------- root_group_properties.hpp --- SYCL root_group properties -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

struct use_root_sync_key {
  using value_t = property_value<use_root_sync_key>;
};

inline constexpr use_root_sync_key::value_t use_root_sync;

template <> struct is_property_key<use_root_sync_key> : std::true_type {};

namespace detail {

template <> struct PropertyToKind<use_root_sync_key> {
  static constexpr PropKind Kind = PropKind::UseRootSync;
};

template <> struct IsCompileTimeProperty<use_root_sync_key> : std::true_type {};

// use_root_group does not need a specialization of PropertyMetaInfo as only the
// runtime needs to know of its existance.

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
