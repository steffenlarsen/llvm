// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -S \
// RUN:    -emit-llvm %s -o - | FileCheck %s

// Tests the optional filter parameter of
// __sycl_detail__::add_ir_attributes_function

#include "mock_properties.hpp"
#include "sycl.hpp"

template<typename ...Properties>
class KernelFunctor {
 public:
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_function(
    {"Prop3", "Prop5"},
    Properties::name..., Properties::value...
    )]]
#endif
  void operator()() const {}
};

int main() {
  sycl::queue q;
  auto f = [=]() {};
  q.submit([&](sycl::handler &h) {
    KernelFunctor<prop5, prop1, prop3, prop4> f{};
    h.single_task<class test_kernel>(f);
  });
}

// CHECK-NOT: "Prop1"="Property string"
// CHECK-NOT: "Prop2"="1"
// CHECK: "Prop3"="true"
// CHECK-NOT: "Prop4"="2"
// CHECK: "Prop5"
// CHECK-NOT: "Prop6"
