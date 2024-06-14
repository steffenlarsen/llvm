//==------- work_group_static.hpp - SYCL work_group_static extension -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/pointers.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

#ifdef __SYCL_DEVICE_ONLY__
// Request a fixed-size allocation in local address space at kernel scope.
extern "C" __DPCPP_SYCL_EXTERNAL __attribute__((opencl_local)) std::uint8_t *
__sycl_allocateLocalMemory(std::size_t Size, std::size_t Alignment);

// Request the pointer to a chunk of local memory, the size of which is
// specified by the kernel launch.
extern "C" __DPCPP_SYCL_EXTERNAL __attribute__((opencl_local)) std::uint8_t *
__sycl_getDynamicLocalMemory(std::size_t Alignment);
#endif

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_HOST_NOT_SUPPORTED(Op)
#else
#define __SYCL_HOST_NOT_SUPPORTED(Op)                                          \
  throw sycl::exception(                                                       \
      sycl::make_error_code(sycl::errc::feature_not_supported),                \
      Op " is not supported on host device.");
#endif

namespace sycl {
inline namespace _V1 {
namespace detail {

// is_unbounded_array is a C++20 trait, so we define it here for now.
template <typename T> struct is_unbounded_array : std::false_type {};
template <typename T> struct is_unbounded_array<T[]> : std::true_type {};

template <typename T>
constexpr bool is_unbounded_array_v = is_unbounded_array<T>::value;

} // namespace detail

namespace ext::oneapi::experimental {

template <typename T>
class
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::global_variable_allowed]]
#endif
    __SYCL_TYPE(work_group_static) work_group_static {

public:
  static_assert(
      std::is_trivially_constructible_v<T> ||
          (sycl::detail::is_unbounded_array_v<T> &&
           std::is_trivially_constructible_v<std::remove_all_extents_t<T>>),
      "Type T must be trivially constructible or an unbounded array with a "
      "trivially constructible element type.");

  static_assert(
      std::is_trivially_destructible_v<T> ||
          (sycl::detail::is_unbounded_array_v<T> &&
           std::is_trivially_destructible_v<std::remove_all_extents_t<T>>),
      "Type T must be trivially destructible or an unbounded array with a "
      "trivially destructible element type.");

  work_group_static() = default;
  work_group_static(const work_group_static &) = delete;
  work_group_static(work_group_static &&) = delete;
  work_group_static &operator=(const work_group_static &) = delete;
  work_group_static &operator=(work_group_static &&) = delete;

  operator T &() const noexcept {
    __SYCL_HOST_NOT_SUPPORTED("Implicit conversion")
    return *Ptr;
  }

  T *operator&() const noexcept {
    __SYCL_HOST_NOT_SUPPORTED("& operator")
    return Ptr;
  }

  template <typename RelayT = T,
            typename = std::enable_if_t<!std::is_array_v<RelayT> &&
                                        std::is_same_v<RelayT, T>>>
  const work_group_static &operator=(const T &Value) const noexcept {
    __SYCL_HOST_NOT_SUPPORTED("Assignment operator")
    *Ptr = Value;
    return this;
  }

protected:
#ifdef __SYCL_DEVICE_ONLY__

  static __attribute__((opencl_local)) T * GetMemory() {
    if constexpr (sycl::detail::is_unbounded_array_v<T>) {
      return reinterpret_cast<__attribute__((opencl_local)) T *>(
          __sycl_getDynamicLocalMemory(alignof(T)));
    } else {
      return reinterpret_cast<__attribute__((opencl_local)) T *>(
          __sycl_allocateLocalMemory(sizeof(T), alignof(T)));
    }
  }
  
  __attribute__((opencl_local)) T *Ptr = GetMemory();
#else
  T *Ptr = nullptr;
#endif
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

#undef __SYCL_HOST_NOT_SUPPORTED
