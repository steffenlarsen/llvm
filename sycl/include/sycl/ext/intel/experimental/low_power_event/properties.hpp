//==--------------- SYCL low power event control property ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/experimental/low_power_event/interrupt_id.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {

struct low_power_event
    : sycl::ext::oneapi::experimental::detail::run_time_property_key<
          sycl::ext::oneapi::experimental::detail::PropKind::LowPowerEvent> {
  low_power_event(interrupt_id ID) : value{ID} {}
  low_power_event() : value{std::nullopt} {}

  const std::optional<interrupt_id> value;
};

using low_power_event_key = low_power_event;

inline bool operator==(const low_power_event &LHS, const low_power_event &RHS) {
  return LHS.value == RHS.value;
}
inline bool operator!=(const low_power_event &LHS, const low_power_event &RHS) {
  return !(LHS == RHS);
}

} // namespace ext::intel::experimental
} // namespace _V1
} // namespace sycl
