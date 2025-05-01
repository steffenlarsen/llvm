//==---------- assert.cpp --- Check assert helpers enqueue -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
 * The positive test here checks that assert fallback assert feature works well.
 * According to the doc, when assert is triggered on device host application
 * should abort. That said, a standard `abort()` function is to be called. The
 * function makes sure the app terminates due `SIGABRT` signal. This makes it
 * impossible to verify the feature in uni-process environment. Hence, we employ
 * multi-process envirnment i.e. we call a `fork()`. The child process is should
 * abort and the parent process verifies it and checks that child prints correct
 * error message to `stderr`. Verification of `stderr` output is performed via
 * pipe.
 */

#include "ur_mock_helpers.hpp"
// Enable use of interop kernel c-tor
#define __SYCL_INTERNAL_API
#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#ifndef _WIN32
#include <sys/ioctl.h>
#include <unistd.h>
#endif // _WIN32

class TestKernel;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<TestKernel> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "TestKernel"; }
};

static constexpr const kernel_param_desc_t Signatures[] = {
    {kernel_param_kind_t::kind_accessor, 4062, 0}};

} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage generateDefaultImage() {
  using namespace sycl::unittest;

  static const std::string KernelName = "TestKernel";

  MockPropertySet PropSet;

  setKernelUsesAssert({KernelName}, PropSet);

  std::vector<MockOffloadEntry> Entries = makeEmptyKernels({KernelName});

  MockDeviceImage Img(std::move(Entries), std::move(PropSet));

  return Img;
}

sycl::unittest::MockDeviceImage Imgs[] = {generateDefaultImage()};
sycl::unittest::MockDeviceImageArray<1> ImgArray{Imgs};

static constexpr int KernelLaunchCounterBase = 0;
static constexpr int MemoryMapCounterBase = 1000;
static int MemoryMapCounter = MemoryMapCounterBase;
#ifndef _WIN32
static int KernelLaunchCounter = KernelLaunchCounterBase;
static constexpr int PauseWaitOnIdx = KernelLaunchCounterBase + 1;
#endif

// Mock redifinitions
static ur_result_t redefinedKernelGetGroupInfoAfter(void *pParams) {
  auto params = *static_cast<ur_kernel_get_group_info_params_t *>(pParams);
  if (*params.ppropName == UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE) {
    if (*params.ppPropSizeRet) {
      **params.ppPropSizeRet = 3 * sizeof(size_t);
    } else if (*params.ppPropValue) {
      auto size = static_cast<size_t *>(*params.ppPropValue);
      size[0] = 1;
      size[1] = 1;
      size[2] = 1;
    }
  }

  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedEventWaitNegative(void *pParams) {
  auto params = *static_cast<ur_enqueue_events_wait_params_t *>(pParams);
  // For negative tests we do not expect the copier kernel to be used, so
  // instead we accept whatever amount we get.
  // This output here is to reduce amount of time requried to debug/reproduce
  // a failing test upon feature break
  printf("Waiting for %i events ", *params.pnumEventsInWaitList);
  for (size_t I = 0; I < *params.pnumEventsInWaitList; ++I)
    printf("%i, ", reinterpret_cast<int *>(*params.pphEvent[I])[0]);
  printf("\n");
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedEnqueueMemBufferMapAfter(void *pParams) {
  auto params = *static_cast<ur_enqueue_mem_buffer_map_params_t *>(pParams);
  MemoryMapCounter++;
  // This output here is to reduce amount of time requried to debug/reproduce a
  // failing test upon feature break
  printf("Memory map %i\n", MemoryMapCounter);

  return UR_RESULT_SUCCESS;
}

namespace TestInteropKernel {
const sycl::context *Context = nullptr;
const sycl::device *Device = nullptr;
int KernelLaunchCounter = ::KernelLaunchCounterBase;

static ur_result_t redefinedKernelGetInfo(void *pParams) {
  auto params = *static_cast<ur_kernel_get_info_params_t *>(pParams);
  if (UR_KERNEL_INFO_CONTEXT == *params.ppropName) {
    ur_context_handle_t UrContext =
        sycl::detail::getSyclObjImpl(*Context)->getHandleRef();

    if (*params.ppPropValue)
      memcpy(*params.ppPropValue, &UrContext, sizeof(UrContext));
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(UrContext);

    return UR_RESULT_SUCCESS;
  }

  if (UR_KERNEL_INFO_PROGRAM == *params.ppropName) {
    ur_program_handle_t URProgram =
        mock::createDummyHandle<ur_program_handle_t>();

    if (*params.ppPropValue)
      memcpy(*params.ppPropValue, &URProgram, sizeof(URProgram));
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(URProgram);

    return UR_RESULT_SUCCESS;
  }

  if (UR_KERNEL_INFO_FUNCTION_NAME == *params.ppropName) {
    static const char FName[] = "TestFnName";
    if (*params.ppPropValue) {
      size_t L = strlen(FName) + 1;
      if (L < *params.ppropSize)
        L = *params.ppropSize;

      memcpy(*params.ppPropValue, FName, L);
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = strlen(FName) + 1;

    return UR_RESULT_SUCCESS;
  }

  return UR_RESULT_ERROR_UNKNOWN;
}

static ur_result_t redefinedEnqueueKernelLaunch(void *pParms) {
  int Val = KernelLaunchCounter++;
  // This output here is to reduce amount of time requried to debug/reproduce a
  // failing test upon feature break
  printf("Enqueued %i\n", Val);

  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedProgramGetInfo(void *pParams) {
  auto params = *static_cast<ur_program_get_info_params_t *>(pParams);
  if (UR_PROGRAM_INFO_NUM_DEVICES == *params.ppropName) {
    static const int V = 1;

    if (*params.ppPropValue)
      memcpy(*params.ppPropValue, &V, sizeof(V));
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(V);

    return UR_RESULT_SUCCESS;
  }

  if (UR_PROGRAM_INFO_DEVICES == *params.ppropName) {
    EXPECT_EQ(*params.ppropSize, 1 * sizeof(ur_device_handle_t));

    ur_device_handle_t Dev = sycl::detail::getSyclObjImpl(*Device)->getHandleRef();

    if (*params.ppPropValue)
      memcpy(*params.ppPropValue, &Dev, sizeof(Dev));
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(Dev);

    return UR_RESULT_SUCCESS;
  }

  // Required if program cache eviction is enabled.
  if (UR_PROGRAM_INFO_BINARY_SIZES == *params.ppropName) {
    size_t BinarySize = 1;

    if (*params.ppPropValue)
      memcpy(*params.ppPropValue, &BinarySize, sizeof(size_t));
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(size_t);

    return UR_RESULT_SUCCESS;
  }

  return UR_RESULT_ERROR_UNKNOWN;
}

static ur_result_t redefinedProgramGetBuildInfo(void *pParams) {
  auto params = *static_cast<ur_program_get_build_info_params_t *>(pParams);
  if (UR_PROGRAM_BUILD_INFO_BINARY_TYPE == *params.ppropName) {
    static const ur_program_binary_type_t T = UR_PROGRAM_BINARY_TYPE_EXECUTABLE;
    if (*params.ppPropValue)
      memcpy(*params.ppPropValue, &T, sizeof(T));
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(T);
    return UR_RESULT_SUCCESS;
  }

  if (UR_PROGRAM_BUILD_INFO_OPTIONS == *params.ppropName) {
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = 0;
    return UR_RESULT_SUCCESS;
  }

  return UR_RESULT_ERROR_UNKNOWN;
}

} // namespace TestInteropKernel

static void setupMockForInterop(sycl::unittest::UrMock<> &Mock,
                                const sycl::context &Ctx,
                                const sycl::device &Dev) {
  using namespace sycl::detail;

  TestInteropKernel::KernelLaunchCounter = ::KernelLaunchCounterBase;
  TestInteropKernel::Device = &Dev;
  TestInteropKernel::Context = &Ctx;

  mock::getCallbacks().set_after_callback("urKernelGetGroupInfo",
                                          &redefinedKernelGetGroupInfoAfter);
  mock::getCallbacks().set_before_callback(
      "urEnqueueKernelLaunch",
      &TestInteropKernel::redefinedEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback("urEnqueueMemBufferMap",
                                          &redefinedEnqueueMemBufferMapAfter);
  mock::getCallbacks().set_before_callback("urEventWait",
                                           &redefinedEventWaitNegative);
  mock::getCallbacks().set_before_callback(
      "urKernelGetInfo", &TestInteropKernel::redefinedKernelGetInfo);
  mock::getCallbacks().set_before_callback(
      "urProgramGetInfo", &TestInteropKernel::redefinedProgramGetInfo);
  mock::getCallbacks().set_before_callback(
      "urProgramGetBuildInfo",
      &TestInteropKernel::redefinedProgramGetBuildInfo);
}

TEST(Assert, TestInteropKernelNegative) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};

  setupMockForInterop(Mock, Ctx, Dev);

  sycl::queue Queue{Ctx, Dev};

  auto URKernel = mock::createDummyHandle<ur_kernel_handle_t>();

  // TODO use make_kernel. This requires a fix in backend.cpp to get adapter
  // from context instead of free getAdapter to allow for mocking of its
  // methods
  sycl::kernel KInterop((cl_kernel)URKernel, Ctx);

  Queue.submit([&](sycl::handler &H) { H.single_task(KInterop); });

  EXPECT_EQ(TestInteropKernel::KernelLaunchCounter,
            KernelLaunchCounterBase + 1);
}

TEST(Assert, TestInteropKernelFromProgramNegative) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};

  setupMockForInterop(Mock, Ctx, Dev);

  sycl::queue Queue{Ctx, Dev};

  sycl::kernel_bundle Bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx);
  sycl::kernel KOrig = Bundle.get_kernel(sycl::get_kernel_id<TestKernel>());

  cl_kernel CLKernel = sycl::get_native<sycl::backend::opencl>(KOrig);
  sycl::kernel KInterop{CLKernel, Ctx};

  Queue.submit([&](sycl::handler &H) { H.single_task(KInterop); });

  EXPECT_EQ(TestInteropKernel::KernelLaunchCounter,
            KernelLaunchCounterBase + 1);
}
