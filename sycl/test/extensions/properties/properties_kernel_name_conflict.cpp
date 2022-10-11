// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;

  auto F = []() {};
  Q.single_task(F);
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::sub_group_size<1>},
      F);

  return 0;
}
