// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

#include <CL/sycl.hpp>
using namespace sycl;
using namespace sycl::ext::codeplay;

constexpr size_t N = 1024;

bool test0(queue q) {
  int A[N];
  for (size_t i = 0; i < N; ++i) {
    A[i] = i;
  }

  {
    buffer<int> aBuff{A, range{N}};
    q.submit([&](handler &cgh) {
      auto aView = item_view<int, 1, access::mode::read_write>(aBuff, cgh);

      cgh.parallel_for<class add_one>(nd_range(range{N}, range{64}), aView,
                                      [=](int &a) { a += 1; });
    });
  }

  for (size_t i = 0; i < N; ++i) {
    if (A[i] != i + 1) {
      return false;
    }
  }

  return true;
}

bool test1(queue q) {
  int A[N];
  for (size_t i = 0; i < N; ++i) {
    A[i] = i;
  }

  int res = 0;

  {
    buffer<int> aBuff{A, range{N}};
    buffer<int> resBuff{&res, range{1}};
    q.submit([&](handler &cgh) {
      auto resAcc = resBuff.get_access<access::mode::read_write>(cgh);
      auto aView = item_view<int, 1, access::mode::read>(aBuff, cgh);

      auto sumReduction = reduction(resAcc, plus<>());
      cgh.parallel_for<class fact>(
          nd_range(range{N}, range{64}), aView, sumReduction,
          [=](const int &a, auto &sum) { sum.combine(a); });
    });
  }

  int expectedRes = 0;
  for (size_t i = 0; i < N; ++i) {
    expectedRes += A[i];
  }

  return expectedRes == res;
}

bool test2(queue q) {
  int A[N];
  int B[N];
  int C[N];
  for (size_t i = 0; i < N; ++i) {
    A[i] = i;
    B[i] = i;
  }

  {
    buffer<int> aBuff{A, range{N}};
    buffer<int> bBuff{B, range{N}};
    buffer<int> cBuff{C, range{N}};
    q.submit([&](handler &cgh) {
      auto aView = item_view<int, 1, access::mode::read>(aBuff, cgh);
      auto bView = item_view<int, 1, access::mode::read>(bBuff, cgh);
      auto cView = item_view<int, 1, access::mode::write>(cBuff, cgh);

      cgh.parallel_for<class vec_add>(
          nd_range(range{N}, range{64}), aView, bView, cView,
          [=](const int &a, const int &b, int &c) { c = a + b; });
    });
  }

  for (size_t i = 0; i < N; ++i) {
    if (C[i] != i + i) {
      return false;
    }
  }
  return true;
}

int main() {
  queue q;
  assert(test0(q) && "Test 0 failed");
  assert(test1(q) && "Test 1 failed");
  assert(test2(q) && "Test 2 failed");
  std::cout << "Test passed." << std::endl;
}
