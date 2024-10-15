//==--------------- SYCL low power event control property ------------------==//
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

struct low_power_event_key
    : sycl::ext::oneapi::experimental::detail::compile_time_property_key<
          sycl::ext::oneapi::experimental::detail::PropKind::LowPowerEvent> {
  using value_t = oneapi::experimental::property_value<low_power_event_key>;
};

inline constexpr low_power_event_key::value_t low_power_event;

} // namespace ext::intel::experimental
} // namespace _V1
} // namespace sycl
