//==- virtual_mem.cpp - sycl_ext_oneapi_virtual_mem virtual mem free funcs -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::experimental {

__SYCL_EXPORT size_t get_minimum_mem_granularity(const device &SyclDevice,
                                                 const context &SyclContext) {
  if (!SyclDevice.has(aspect::ext_oneapi_virtual_mem))
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device does not support aspect::ext_oneapi_virtual_mem.");

  std::shared_ptr<sycl::detail::device_impl> DeviceImpl =
      sycl::detail::getSyclObjImpl(SyclDevice);
  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(SyclContext);
  const sycl::detail::plugin &Plugin = ContextImpl->getPlugin();
#ifndef NDEBUG
  size_t InfoOutputSize;
  Plugin.call<sycl::detail::PiApiKind::piextVirtualMemGranularityGetInfo>(
      ContextImpl->getHandleRef(), DeviceImpl->getHandleRef(),
      PI_EXT_ONEAPI_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM, 0, nullptr,
      &InfoOutputSize);
  assert(InfoOutputSize == sizeof(size_t) &&
         "Unexpected output size of granularity info query.");
#endif // NDEBUG
  size_t Granularity;
  Plugin.call<sycl::detail::PiApiKind::piextVirtualMemGranularityGetInfo>(
      ContextImpl->getHandleRef(), DeviceImpl->getHandleRef(),
      PI_EXT_ONEAPI_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM, sizeof(size_t),
      &Granularity, nullptr);
  return Granularity;
}

__SYCL_EXPORT size_t get_recommended_mem_granularity(
    const device &SyclDevice, const context &SyclContext) {
  if (!SyclDevice.has(aspect::ext_oneapi_virtual_mem))
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device does not support aspect::ext_oneapi_virtual_mem.");

  std::shared_ptr<sycl::detail::device_impl> DeviceImpl =
      sycl::detail::getSyclObjImpl(SyclDevice);
  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(SyclContext);
  const sycl::detail::plugin &Plugin = ContextImpl->getPlugin();
#ifndef NDEBUG
  size_t InfoOutputSize;
  Plugin.call<sycl::detail::PiApiKind::piextVirtualMemGranularityGetInfo>(
      ContextImpl->getHandleRef(), DeviceImpl->getHandleRef(),
      PI_EXT_ONEAPI_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED, 0, nullptr,
      &InfoOutputSize);
  assert(InfoOutputSize == sizeof(size_t) &&
         "Unexpected output size of granularity info query.");
#endif // NDEBUG
  size_t Granularity;
  Plugin.call<sycl::detail::PiApiKind::piextVirtualMemGranularityGetInfo>(
      ContextImpl->getHandleRef(), DeviceImpl->getHandleRef(),
      PI_EXT_ONEAPI_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED, sizeof(size_t),
      &Granularity, nullptr);
  return Granularity;
}

} // Namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // Namespace sycl
