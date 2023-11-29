//==-- nd_reduction.hpp - ONEAPI multi-dimensional reductions --*- C++ --*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/reduction.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi {

/// Constructs a reduction object using the given buffer \p Var, handler \p CGH,
/// reduction operation \p Combiner, and optional reduction properties.
/// The reduction algorithm may be less efficient if the specified binary
/// operation does not have a known identity.
template <typename T, int Dimensions, typename AllocatorT,
          typename BinaryOperation>
auto reduction(buffer<T, Dimensions, AllocatorT> Var, handler &CGH,
               BinaryOperation Combiner, const property_list &PropList = {}) {
  std::ignore = CGH;
  bool InitializeToIdentity =
      PropList.has_property<sycl::property::reduction::initialize_to_identity>();
  return sycl::detail::make_reduction<BinaryOperation, Dimensions, 1, false>(
      Var, Combiner, InitializeToIdentity);
}

/// Constructs a reduction object using the given buffer \p Var, handler \p CGH,
/// reduction identity value \p Identity, reduction operation \p Combiner,
/// and optional reduction properties.
template <typename T, int Dimensions, typename AllocatorT,
          typename BinaryOperation>
auto reduction(buffer<T, Dimensions, AllocatorT> Var, handler &CGH,
               const T &Identity, BinaryOperation Combiner,
               const property_list &PropList = {}) {
  std::ignore = CGH;
  bool InitializeToIdentity =
      PropList.has_property<sycl::property::reduction::initialize_to_identity>();
  return sycl::detail::make_reduction<BinaryOperation, Dimensions, 1, true>(
      Var, Identity, Combiner, InitializeToIdentity);
}

} // namespace ext::oneapi
} // namespace _V1
} // namespace sycl