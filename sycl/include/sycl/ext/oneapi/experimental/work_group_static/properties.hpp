//==----- properties.hpp - SYCL properties associated with device_global ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#include <cstdint>
#include <iosfwd>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

struct work_group_static_size
    : detail::run_time_property_key<detail::PropKind::WorkGroupSpecificSize> {
  constexpr work_group_static_size(size_t Bytes) : value(Bytes) {}
  size_t value;
};

using work_group_static_size_key = work_group_static_size;

template <>
struct is_property_key<work_group_static_size_key> : std::true_type {};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
