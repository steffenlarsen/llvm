// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

namespace oneapiexp = sycl::ext::oneapi::experimental;
namespace oneapi = sycl::ext::oneapi;

template <typename PackT> struct PackInfo;
template <size_t NumElems, typename StorageT>
struct PackInfo<oneapiexp::int4_packed<NumElems, StorageT>> {
  static constexpr size_t num_elems = NumElems;
  using storage_t = StorageT;
};
template <size_t NumElems, typename StorageT>
struct PackInfo<oneapiexp::uint4_packed<NumElems, StorageT>> {
  static constexpr size_t num_elems = NumElems;
  using storage_t = StorageT;
};

template <typename ToT, typename FromT> ToT GenDownConvert() {
  using MarrayT = sycl::marray<FromT, PackInfo<ToT>::num_elems>;
  return ToT{MarrayT{FromT{}}};
}

template <typename ToT, typename FromT>
sycl::marray<ToT, PackInfo<FromT>::num_elems> GenUpConvert() {
  return {FromT{}};
}

SYCL_EXTERNAL auto Conversions() {
  return std::make_tuple(
      // CHECK-DAG: i8 @_Z38__builtin_spirv_ConvertHF16ToInt4INTELDv2_DF16_(<2 x half>
      GenDownConvert<oneapiexp::int4_packed<16, uint8_t>, sycl::half>(),
      // CHECK-DAG: i32 @_Z38__builtin_spirv_ConvertHF16ToInt4INTELDv8_DF16_(<8 x half>
      GenDownConvert<oneapiexp::int4_packed<16, uint32_t>, sycl::half>(),
      // CHECK-DAG: i8 @_Z38__builtin_spirv_ConvertBF16ToInt4INTELDv2_t(<2 x i16>
      GenDownConvert<oneapiexp::int4_packed<16, uint8_t>, oneapi::bfloat16>(),
      // CHECK-DAG: i32 @_Z38__builtin_spirv_ConvertBF16ToInt4INTELDv8_t(<8 x i16>
      GenDownConvert<oneapiexp::int4_packed<16, uint32_t>, oneapi::bfloat16>(),

      // CHECK-DAG: <2 x half> @_Z38__builtin_spirv_ConvertInt4ToHF16INTELh(i8
      GenUpConvert<sycl::half, oneapiexp::int4_packed<16, uint8_t>>(),
      // CHECK-DAG: <8 x half> @_Z38__builtin_spirv_ConvertInt4ToHF16INTELj(i32
      GenUpConvert<sycl::half, oneapiexp::int4_packed<16, uint32_t>>(),
      // CHECK-DAG: <2 x i16> @_Z38__builtin_spirv_ConvertInt4ToBF16INTELh(i8
      GenUpConvert<oneapi::bfloat16, oneapiexp::int4_packed<16, uint8_t>>(),
      // CHECK-DAG: <8 x i16> @_Z38__builtin_spirv_ConvertInt4ToBF16INTELj(i32
      GenUpConvert<oneapi::bfloat16, oneapiexp::int4_packed<16, uint32_t>>(),

      // CHECK-DAG: i8 @_Z39__builtin_spirv_ConvertHF16ToUInt4INTELDv2_DF16_(<2 x half>
      GenDownConvert<oneapiexp::uint4_packed<16, uint8_t>, sycl::half>(),
      // CHECK-DAG: i32 @_Z39__builtin_spirv_ConvertHF16ToUInt4INTELDv8_DF16_(<8 x half>
      GenDownConvert<oneapiexp::uint4_packed<16, uint32_t>, sycl::half>(),
      // CHECK-DAG: i8 @_Z39__builtin_spirv_ConvertBF16ToUInt4INTELDv2_t(<2 x i16>
      GenDownConvert<oneapiexp::uint4_packed<16, uint8_t>, oneapi::bfloat16>(),
      // CHECK-DAG: i32 @_Z39__builtin_spirv_ConvertBF16ToUInt4INTELDv8_t(<8 x i16>
      GenDownConvert<oneapiexp::uint4_packed<16, uint32_t>, oneapi::bfloat16>(),

      // CHECK-DAG: <2 x half> @_Z39__builtin_spirv_ConvertUInt4ToHF16INTELh(i8
      GenUpConvert<sycl::half, oneapiexp::uint4_packed<16, uint8_t>>(),
      // CHECK-DAG: <8 x half> @_Z39__builtin_spirv_ConvertUInt4ToHF16INTELj(i32
      GenUpConvert<sycl::half, oneapiexp::uint4_packed<16, uint32_t>>(),
      // CHECK-DAG: <2 x i16> @_Z39__builtin_spirv_ConvertUInt4ToBF16INTELh(i8
      GenUpConvert<oneapi::bfloat16, oneapiexp::uint4_packed<16, uint8_t>>(),
      // CHECK-DAG: <8 x i16> @_Z39__builtin_spirv_ConvertUInt4ToBF16INTELj(i32
      GenUpConvert<oneapi::bfloat16, oneapiexp::uint4_packed<16, uint32_t>>());
}
