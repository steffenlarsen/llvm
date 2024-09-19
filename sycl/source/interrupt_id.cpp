//==------------------------- SYCL interrupt IDs ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/intel/experimental/low_power_event/interrupt_id.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {

interrupt_id make_interrupt_id(
    const backend_input_t<backend::ext_oneapi_level_zero, interrupt_id>
        &BackendObject) {
  return {std::make_shared{BackendObject}};
}

} // namespace ext::intel::experimental
} // namespace _V1
} // namespace sycl
