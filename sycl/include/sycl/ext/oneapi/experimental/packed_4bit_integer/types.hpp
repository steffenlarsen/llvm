//===------------- types.hpp - SYCL 4bit integer type header --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_types.hpp>
#include <sycl/aspects.hpp>
#include <sycl/detail/memcpy.hpp>
#include <sycl/exception.hpp>
#include <sycl/marray.hpp>

#include <array>
#include <cstddef>

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_USES_ASPECT_ON_DEVICE(ASPECT)                                   \
  [[__sycl_detail__::__uses_aspects__(ASPECT)]]
#else // __SYCL_DEVICE_ONLY__
#define __SYCL_USES_ASPECT_ON_DEVICE(ASPECT)
#endif

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_UPCONVERT_COMMON_BODY(TO_TYPE, NUM_ELEMS, NUM_STORAGE_BLOCKS,   \
                                     STORAGE, STORAGE_TYPE)                    \
  constexpr size_t ElemsPerBlock = sizeof(STORAGE_TYPE) * 2;                   \
  using ValRep = std::conditional_t<std::is_same_v<TO_TYPE, sycl::half>,       \
                                    _Float16, uint16_t>;                       \
  using ValRepVec = ValRep __attribute__((ext_vector_type(ElemsPerBlock)));    \
  sycl::marray<TO_TYPE, NUM_ELEMS> Res;                                        \
  for (size_t I = 0; I < NUM_STORAGE_BLOCKS; ++I) {                            \
    auto ConvertedBlock =                                                      \
        static_cast<ValRepVec>(sycl::bit_cast<Int4BitVecT>(STORAGE[I]));       \
    for (size_t J = 0; J < ElemsPerBlock; ++J)                                 \
      Res[I * ElemsPerBlock + J] = sycl::bit_cast<TO_TYPE>(ConvertedBlock[J]); \
  }                                                                            \
  return Res;

#define __SYCL_DOWNCONVERT_COMMON_BODY(VALS, NUM_ELEMS, NUM_STORAGE_BLOCKS,    \
                                       STORAGE, STORAGE_TYPE)                  \
  constexpr size_t ElemsPerBlock = sizeof(STORAGE_TYPE) * 2;                   \
  using ValElem =                                                              \
      std::remove_cv_t<std::remove_reference_t<decltype(VALS[0])>>;            \
  using ValRep = std::conditional_t<std::is_same_v<ValElem, sycl::half>,       \
                                    _Float16, uint16_t>;                       \
  using ValRepVec = ValRep __attribute__((ext_vector_type(ElemsPerBlock)));    \
  for (size_t I = 0; I < NUM_STORAGE_BLOCKS; ++I) {                            \
    ValRepVec InputChunkVec;                                                   \
    for (size_t J = 0; J < ElemsPerBlock; ++J)                                 \
      InputChunkVec[J] = sycl::bit_cast<ValRep>(VALS[I * ElemsPerBlock + J]);  \
    STORAGE[I] =                                                               \
        sycl::bit_cast<STORAGE_TYPE>(static_cast<Int4BitVecT>(InputChunkVec)); \
  }

#else // __SYCL_DEVICE_ONLY__
#define __SYCL_UPCONVERT_COMMON_BODY(TO_TYPE, NUM_ELEMS, NUM_STORAGE_BLOCKS,   \
                                     STORAGE, UNSIGNED_STORAGE_TYPE)           \
  throw sycl::exception(                                                       \
      make_error_code(errc::invalid),                                          \
      "Conversion from 4-bit integer type is not supported on host.");

#define __SYCL_DOWNCONVERT_COMMON_BODY(VALS, NUM_ELEMS, NUM_STORAGE_BLOCKS,    \
                                       STORAGE, UNSIGNED_STORAGE_TYPE)         \
  std::ignore = VALS;                                                          \
  throw sycl::exception(                                                       \
      make_error_code(errc::invalid),                                          \
      "Conversion to 4-bit integer type is not supported on host.");

#endif

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
template <size_t NumElems, typename StorageT = std::conditional_t<
                               NumElems % 8 == 0, uint32_t, uint8_t>>
class __SYCL_USES_ASPECT_ON_DEVICE(aspect::ext_oneapi_int4) int4_packed {
  static_assert(std::is_same_v<StorageT, uint8_t> ||
                    std::is_same_v<StorageT, uint32_t>,
                "StorageT must be either uint8_t or uint32_t.");
  static_assert(NumElems > 0, "Number of packed elements cannot be 0.");

  static constexpr size_t NumStorageBlocks =
      1 + ((NumElems - 1) / sizeof(StorageT));
  StorageT Storage[NumStorageBlocks];

#ifdef __SYCL_DEVICE_ONLY__
  using Int4BitVecT = sycl::detail::ap_int<4> __attribute__((
      ext_vector_type(sizeof(StorageT) * 2)));
#endif

public:
  int4_packed() = default;
  int4_packed(const int4_packed &) = default;
  int4_packed &operator=(const int4_packed &) = default;

  explicit int4_packed(const marray<half, NumElems> &Vals) { assign(Vals); }
  explicit int4_packed(const marray<bfloat16, NumElems> &Vals) { assign(Vals); }
  explicit int4_packed(const marray<int8_t, NumElems> &Vals) { assign(Vals); }
  explicit int4_packed(const marray<uint8_t, NumElems> &Vals) { assign(Vals); }

  int4_packed &operator=(const marray<half, NumElems> &Vals) { assign(Vals); }
  int4_packed &operator=(const marray<bfloat16, NumElems> &Vals) {
    assign(Vals);
  }
  int4_packed &operator=(const marray<int8_t, NumElems> &Vals) { assign(Vals); }
  int4_packed &operator=(const marray<uint8_t, NumElems> &Vals) {
    assign(Vals);
  }

  void assign(const marray<half, NumElems> &Vals) {
    __SYCL_DOWNCONVERT_COMMON_BODY(Vals, NumElems, NumStorageBlocks, Storage,
                                   StorageT)
  }
  void assign(const marray<bfloat16, NumElems> &Vals) {
    __SYCL_DOWNCONVERT_COMMON_BODY(Vals, NumElems, NumStorageBlocks, Storage,
                                   StorageT)
  }
  void assign(const marray<int8_t, NumElems> &Vals) {
    __SYCL_DOWNCONVERT_COMMON_BODY(Vals, NumElems, NumStorageBlocks, Storage,
                                   StorageT)
  }
  void assign(const marray<uint8_t, NumElems> &Vals) {
    __SYCL_DOWNCONVERT_COMMON_BODY(Vals, NumElems, NumStorageBlocks, Storage,
                                   StorageT)
  }

  template <typename TargetStorageT>
  operator int4_packed<NumElems, TargetStorageT>() const {
    int4_packed<NumElems, TargetStorageT> Res;
    sycl::detail::memcpy_no_adl(&Res, Storage, 1 + ((NumElems - 1) / 2));
    return Res;
  }

  operator marray<half, NumElems>() const {
    __SYCL_UPCONVERT_COMMON_BODY(half, NumElems, NumStorageBlocks, Storage,
                                 StorageT)
  }
  operator marray<bfloat16, NumElems>() const {
    __SYCL_UPCONVERT_COMMON_BODY(bfloat16,

                                 NumElems, NumStorageBlocks, Storage, StorageT)
  }
  operator marray<int8_t, NumElems>() const {
    __SYCL_UPCONVERT_COMMON_BODY(int8_t, NumElems, NumStorageBlocks, Storage,
                                 StorageT)
  }
  operator marray<uint8_t, NumElems>() const {
    __SYCL_UPCONVERT_COMMON_BODY(uint8_t, NumElems, NumStorageBlocks, Storage,
                                 StorageT)
  }
};

template <size_t NumElems, typename StorageT = std::conditional_t<
                               NumElems % 8 == 0, uint32_t, uint8_t>>
class __SYCL_USES_ASPECT_ON_DEVICE(aspect::ext_oneapi_int4) uint4_packed {
  static_assert(std::is_same_v<StorageT, uint8_t> ||
                    std::is_same_v<StorageT, uint32_t>,
                "StorageT must be either uint8_t or uint32_t.");
  static_assert(NumElems > 0, "Number of packed elements cannot be 0.");

  static constexpr size_t NumStorageBlocks =
      1 + ((NumElems - 1) / sizeof(StorageT));
  StorageT Storage[NumStorageBlocks];

#ifdef __SYCL_DEVICE_ONLY__
  using Int4BitVecT = sycl::detail::ap_uint<4> __attribute__((
      ext_vector_type(sizeof(StorageT) * 2)));
#endif

public:
  uint4_packed() = default;
  uint4_packed(const uint4_packed &) = default;
  uint4_packed &operator=(const uint4_packed &) = default;

  explicit uint4_packed(const marray<half, NumElems> &Vals) { assign(Vals); }
  explicit uint4_packed(const marray<bfloat16, NumElems> &Vals) {
    assign(Vals);
  }
  explicit uint4_packed(const marray<int8_t, NumElems> &Vals) { assign(Vals); }
  explicit uint4_packed(const marray<uint8_t, NumElems> &Vals) { assign(Vals); }

  uint4_packed &operator=(const marray<half, NumElems> &Vals) { assign(Vals); }
  uint4_packed &operator=(const marray<bfloat16, NumElems> &Vals) {
    assign(Vals);
  }
  uint4_packed &operator=(const marray<int8_t, NumElems> &Vals) {
    assign(Vals);
  }
  uint4_packed &operator=(const marray<uint8_t, NumElems> &Vals) {
    assign(Vals);
  }

  void assign(const marray<half, NumElems> &Vals) {
    __SYCL_DOWNCONVERT_COMMON_BODY(Vals, NumElems, NumStorageBlocks, Storage,
                                   StorageT)
  }
  void assign(const marray<bfloat16, NumElems> &Vals) {
    __SYCL_DOWNCONVERT_COMMON_BODY(Vals, NumElems, NumStorageBlocks, Storage,
                                   StorageT)
  }
  void assign(const marray<int8_t, NumElems> &Vals) {
    __SYCL_DOWNCONVERT_COMMON_BODY(Vals, NumElems, NumStorageBlocks, Storage,
                                   StorageT)
  }
  void assign(const marray<uint8_t, NumElems> &Vals) {
    __SYCL_DOWNCONVERT_COMMON_BODY(Vals, NumElems, NumStorageBlocks, Storage,
                                   StorageT)
  }

  template <typename TargetStorageT>
  operator uint4_packed<NumElems, TargetStorageT>() const {
    uint4_packed<NumElems, TargetStorageT> Res;
    sycl::detail::memcpy_no_adl(&Res, Storage, 1 + ((NumElems - 1) / 2));
    return Res;
  }

  operator marray<half, NumElems>() const {
    __SYCL_UPCONVERT_COMMON_BODY(half, NumElems, NumStorageBlocks, Storage,
                                 StorageT)
  }
  operator marray<bfloat16, NumElems>() const {
    __SYCL_UPCONVERT_COMMON_BODY(bfloat16, NumElems, NumStorageBlocks, Storage,
                                 StorageT)
  }
  operator marray<int8_t, NumElems>() const {
    __SYCL_UPCONVERT_COMMON_BODY(int8_t, NumElems, NumStorageBlocks, Storage,
                                 StorageT)
  }
  operator marray<uint8_t, NumElems>() const {
    __SYCL_UPCONVERT_COMMON_BODY(uint8_t, NumElems, NumStorageBlocks, Storage,
                                 StorageT)
  }
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

#undef __SYCL_USES_ASPECT_ON_DEVICE
#undef __SYCL_DOWNCONVERT_COMMON_BODY
#undef __SYCL_UPCONVERT_COMMON_BODY
