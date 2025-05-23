// REQUIRES: aspect-ext_oneapi_int4, aspect-usm_shared_allocations
// RUN: %{build} -o %t.out

// Test for 4-bit packed integer type conversions between storage types.

#include <sycl/usm.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/packed_4bit_integer/types.hpp>

namespace oneapiexp = sycl::ext::oneapi::experimental;
namespace oneapi = sycl::ext::oneapi;

template <template <size_t, typename> typename T, typename FromStorageT,
          typename ToStorageT>
int Check(sycl::queue &Q, std::string_view TName,
          std::string_view FromStorageTName, std::string_view ToStorageTName) {
  constexpr size_t NumElems = 5;
  int Failed = 0;

  auto *InData = sycl::malloc_shared<sycl::marray<sycl::half, NumElems>>(1, Q);
  auto *OutData = sycl::malloc_shared<sycl::marray<sycl::half, NumElems>>(1, Q);

  float Init = 0.12345678f;
  for (size_t I = 0; I < NumElems; ++I) {
    (*InData)[I] = Init;
    (*OutData)[I] = 0.0f;
    Init *= 10;
  }

  Q.single_task([=]() {
     T<NumElems, FromStorageT> X{*InData};
     T<NumElems, ToStorageT> Y{X};
     *OutData = Y;
   }).wait_and_throw();

  for (size_t I = 0; I < NumElems; ++I) {
    if ((*InData)[I] == (*OutData)[I])
      continue;
    std::cout << "Failed conversion from storage type " << FromStorageTName
              << " to " << ToStorageTName << " for " << TName << std::endl;
    ++Failed;
  }

  sycl::free(InData, Q);
  sycl::free(OutData, Q);
  return Failed;
}

int main() {
  sycl::queue Q;
  int Failed = 0;
  Failed += Check<oneapiexp::int4_packed, uint8_t, uint32_t>(
      Q, "int4_packed", "uint8_t", "uint32_t");
  Failed += Check<oneapiexp::int4_packed, uint32_t, uint8_t>(
      Q, "int4_packed", "uint32_t", "uint8_t");
  Failed += Check<oneapiexp::uint4_packed, uint8_t, uint32_t>(
      Q, "uint4_packed", "uint8_t", "uint32_t");
  Failed += Check<oneapiexp::uint4_packed, uint32_t, uint8_t>(
      Q, "uint4_packed", "uint32_t", "uint8_t");
  return Failed;
}
