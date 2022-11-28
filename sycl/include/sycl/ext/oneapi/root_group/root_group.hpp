//==------------ root_group.hpp --- SYCL root_group group class ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/helpers.hpp>
#include <sycl/id.hpp>
#include <sycl/memory_enums.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/range.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

namespace detail {
// Helper for creating dimensionality-dependendant SYCL classes with repeated
// values.
template <int Dimensions> struct RepeatValues;
template <> struct RepeatValues<1> {
  template <size_t Value> static id<1> RepeatId() { return Value; }
  template <size_t Value> static range<1> RepeatRange() { return Value; }
};
template <> struct RepeatValues<2> {
  template <size_t Value> static id<2> RepeatId() { return {Value, Value}; }
  template <size_t Value> static range<2> RepeatRange() {
    return {Value, Value};
  }
};
template <> struct RepeatValues<3> {
  template <size_t Value> static id<3> RepeatId() {
    return {Value, Value, Value};
  }
  template <size_t Value> static range<3> RepeatRange() {
    return {Value, Value, Value};
  }
};
}

template <int Dimensions> class root_group {
public:
  using id_type = id<Dimensions>;
  using range_type = range<Dimensions>;
  using linear_id_type = size_t;
  static constexpr int dimensions = Dimensions;
  static constexpr memory_scope fence_scope = memory_scope::device;

  root_group() = delete;

  id<Dimensions> get_group_id() const {
    return detail::RepeatValues<Dimensions>::template RepeatId<0>();
  }

  id<Dimensions> get_local_id() const { return MNDItem.get_global_id(); }

  range<Dimensions> get_group_range() const {
    return detail::RepeatValues<Dimensions>::template RepeatRange<1>();
  }

  range<Dimensions> get_local_range() const {
    return MNDItem.get_global_range();
  }

  range<Dimensions> get_max_local_range() const { return get_local_range(); }

  size_t get_group_linear_id() const { return 0; }

  size_t get_local_linear_id() const { return MNDItem.get_global_linear_id(); }

  size_t get_group_linear_range() const { return 1; }

  size_t get_local_linear_range() const {
    return MNDItem.get_global_linear_range();
  }

  bool leader() const { return get_local_linear_id() == 0; }

private:
  nd_item<Dimensions> MNDItem;

  root_group(nd_item<Dimensions> NDItem) : MNDItem{NDItem} {}

  friend class sycl::detail::Builder;
};

namespace this_kernel {
template <int Dimensions> root_group<Dimensions> get_root_group() {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::Builder::getElement(
      sycl::detail::declptr<root_group<Dimensions>>());
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host");
#endif
}
} // namespace this_kernel

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
