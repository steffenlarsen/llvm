// REQUIRES: aspect-ext_oneapi_int4, aspect-usm_shared_allocations
// RUN: %{build} -o %t.out

// Test for conversion to and from packed 4-bit signed integer types.

#include <sycl/usm.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/packed_4bit_integer/types.hpp>

namespace oneapiexp = sycl::ext::oneapi::experimental;
namespace oneapi = sycl::ext::oneapi;

static constexpr size_t NumVals = 16;
static constexpr int8_t Vals[NumVals] = {-8, -7, -6, -5, -4, -3, -2, -1,
                                         0,  1,  2,  3,  4,  5,  6,  7};

template <typename T1, typename T2>
int check(T1 Actual, T2 Expected, size_t I) {
  if (Actual == Expected)
    return 0;
  std::cout << "Unexpected result for index " << I << ": " << Actual
            << " != " << Expected;
  return 1;
}

template <typename PackedT, typename InT> int test(sycl::queue &Q) {
  int Failed = 0;

  auto *OutHalf = sycl::malloc_shared<sycl::marray<sycl::half, NumVals>>(1, Q);
  auto *OutBF16 =
      sycl::malloc_shared<sycl::marray<oneapi::bfloat16, NumVals>>(1, Q);

  OutHalf[0] = sycl::marray<sycl::half, NumVals>{0.0f};
  OutBF16[0] = sycl::marray<oneapi::bfloat16, NumVals>{0.0f};

  Q.single_task([=]() {
     sycl::marray<InT, NumVals> InVals{0};
     for (size_t I = 0; I < NumVals; ++I)
       InVals[I] = Vals[I];
     PackedT Pack{InVals};

     OutHalf[0] = static_cast<sycl::marray<sycl::half, NumVals>>(Pack);
     OutBF16[0] = static_cast<sycl::marray<oneapi::bfloat16, NumVals>>(Pack);
   }).wait();

  for (size_t I = 0; I < NumVals; ++I) {
    check(static_cast<int16_t>(OutHalf[0][I]), Vals[I], I);
    check(static_cast<int16_t>(OutBF16[0][I]), Vals[I], I);
  }

  sycl::free(OutHalf, Q);
  sycl::free(OutBF16, Q);

  return Failed;
}

int main() {
  int Failed = 0;

  sycl::queue Q;

  Failed += test<oneapiexp::int4_packed<NumVals, uint8_t>, sycl::half>(Q);
  Failed += test<oneapiexp::int4_packed<NumVals, uint8_t>, oneapi::bfloat16>(Q);
  Failed += test<oneapiexp::int4_packed<NumVals, uint8_t>, int8_t>(Q);
  Failed += test<oneapiexp::int4_packed<NumVals, uint8_t>, uint8_t>(Q);
  Failed += test<oneapiexp::int4_packed<NumVals, uint32_t>, sycl::half>(Q);
  Failed +=
      test<oneapiexp::int4_packed<NumVals, uint32_t>, oneapi::bfloat16>(Q);
  Failed += test<oneapiexp::int4_packed<NumVals, uint32_t>, int8_t>(Q);
  Failed += test<oneapiexp::int4_packed<NumVals, uint32_t>, uint8_t>(Q);

  return Failed;
}
