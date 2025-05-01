# Assert feature

**IMPORTANT**: This document is a draft.

Using the standard C++ `assert` API ("assertions") is an important debugging
technique widely used by developers. This document describes the design of
supporting assertions within SYCL device code.
The basic approach we chose is delivering device-side assertions as call to
`std::abort()` at host-side.

As usual, device-side assertions can be disabled by defining `NDEBUG` macro at
compile time.

## Use-case example

```c++
#include <cassert>
#include <sycl/sycl.hpp>

using namespace sycl;

void user_func(item<2> Item) {
  assert((Item[0] % 2) && “Nil”);
}

int main() {
  queue Q;
  Q.submit([&] (handler& CGH) {
    CGH.parallel_for<class TheKernel>(range<2>{N, M}, [=](item<2> It) {
      do_smth();
      user_func(It);
      do_smth_else();
    });
  });
  Q.wait();
  std::cout << “One shouldn’t see this message.“;
  return 0;
}
```

In this use-case every work-item with even index along 0 dimension will trigger
assertion failure. Assertion failure should trigger a call to `std::abort()` at
host as described in
[extension](../extensions/supported/sycl_ext_oneapi_assert.asciidoc).
Even though multiple failures of the same or different assertions can happen in
multiple work-items, implementation is required to deliver at least one
assertion. The assertion failure message is printed to `stderr` by DPCPP
Runtime or underlying backend.

When multiple kernels are enqueued and more than one fail at assertion, at least
one assertion should be reported.


## User requirements

From user's point of view there are the following requirements:

| # | Title | Description | Importance |
| - | ----- | ----------- | ---------- |
| 1 | Abort DPC++ application | Abort host application when assert function is called and print a message about assertion | Must have |
| 2 | Print assert message | Assert function should print message to stderr at host | Must have |
| 3 | Stop under debugger | When debugger is attached, break at assertion point | Highly desired |
| 4 | Reliability | Assert failure should be reported regardless of kernel deadlock | Highly desired |


## Terms

 - Device-side Runtime - runtime library supplied by the Native Device Compiler
   and running on the device.
 - Native Device Compiler - compiler which generates device-native binary image
   based on input SPIR-V image.
 - Low-level Runtime - the backend/runtime behind DPCPP Runtime accessed via
   Unified Runtime.


## How it works?

`assert(expr)` macro ends up in call to `__devicelib_assert_fail`. This function
is part of [Device library extension](https://github.com/intel/llvm/blob/sycl/doc/design/DeviceLibExtensions.rst#cl_intel_devicelib_cassert).

The format of the assert message is unspecified, but it will always include the
text of the failing expression, the values of the standard macros `__FILE__` and
`__LINE__`, and the value of the standard variable `__func__`. If the failing
assert comes from an `nd_range` `parallel_for` it will also include the global
ID and the local ID of the failing work item.

Implementation of this function is supplied by the Native Device Compiler.


## Implementation

The implementation guarantees assertion failure notification delivery to the
host regardless of kernel behavior which hit the assertion. If backend
suports this, it must report support for the "cl_intel_devicelib_assert"
device extension.

The Native Device Compiler is responsible for providing implementation of
`__devicelib_assert_fail` which completely hides details of communication
between the device code and the Low-Level Runtime from the SYCL device compiler
and runtime. The Low-Level Runtime is responsible for:
 - detecting if assert failure took place;
 - flushing assert message to `stderr` on host.

The following sequence of events describes how user code gets notified:
 - Device side:
   1. Assert fails in device-code in kernel
      // It's not defined if GPU thread stops execution
      // Other GPU threads are left untouched
   2. Specialized version of `__devicelib_assert_fail` is called
   3. Device immediately signals to host (Low-Level Runtime)
 - Host side:
   1. The assert failure gets detected by Low-Level Runtime
   2. Low-Level Runtime prints assert failure message to `stderr`
   3. Low-Level Runtime calls `abort()`

