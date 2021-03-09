//==--------------- item_view.hpp - SYCL item view -------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/accessor.hpp>

#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declarations
template <typename, int, typename, typename> class buffer;
class handler;

namespace ext {
namespace codeplay {
namespace detail {

/// Base non-template class which is a base class for all item views
/// classes. It is needed to detect the item views.
class item_view_base {};

/// Predicate returning true if template type parameter is an item view.
template <typename T> struct IsItemView {
  static constexpr bool value = std::is_base_of<item_view_base, T>::value;
};

} // namespace detail

template <typename DataT, int Dimensions, access::mode AccessMode>
class __SYCL_EXPORT item_view : private detail::item_view_base {
public:
  using value_type = DataT;
  using reference = typename std::conditional<AccessMode == access::mode::read,
                                              const DataT &, DataT &>::type;
  static constexpr access::mode accessor_mode = AccessMode;
  static constexpr int dimensions = Dimensions;

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT>
  item_view(buffer<T, Dims, AllocatorT> &BufferRef,
            handler &CommandGroupHandlerRef, const property_list &PropList = {})
      : access{BufferRef, CommandGroupHandlerRef} {}

private:
  using accessor_type =
      accessor<DataT, Dimensions, AccessMode, target::global_buffer>;
  accessor_type access;

  friend class sycl::handler;
};

} // namespace codeplay
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
