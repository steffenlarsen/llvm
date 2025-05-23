= sycl_ext_oneapi_packed_4bit_integer

:source-highlighter: coderay
:coderay-linenums-mode: table

// This section needs to be after the document title.
:doctype: book
:toc2:
:toc: left
:encoding: utf-8
:lang: en
:dpcpp: pass:[DPC++]
:endnote: &#8212;{nbsp}end{nbsp}note

// Set the default source code type in this document to C++,
// for syntax highlighting purposes.  This is needed because
// docbook uses c++ and html5 uses cpp.
:language: {basebackend@docbook:c++:cpp}


== Notice

[%hardbreaks]
Copyright (C) 2024 Intel Corporation.  All rights reserved.

Khronos(R) is a registered trademark and SYCL(TM) and SPIR(TM) are trademarks
of The Khronos Group Inc.  OpenCL(TM) is a trademark of Apple Inc. used by
permission by Khronos.


== Contact

To report problems with this extension, please open a new issue at:

https://github.com/intel/llvm/issues


== Dependencies

This extension is written against the SYCL 2020 revision 8 specification.  All
references below to the "core SYCL specification" or to section numbers in the
SYCL specification refer to that revision.


== Status

This is a proposed extension specification, intended to gather community
feedback.  Interfaces defined in this specification may not be implemented yet
or may be in a preliminary state.  The specification itself may also change in
incompatible ways before it is finalized.  *Shipping software products should
not rely on APIs defined in this specification.*


== Backend support status

The `int4_packed` and `uint4_packed` types from this extension may only be used
on a device that has `aspect::ext_oneapi_int4`.  The application must check that
the device has this aspect before submitting a kernel using these types in this
extension.  If the application fails to do this, the implementation throws
a synchronous exception with the `errc::kernel_not_supported` error code
when the kernel is submitted to the queue.  Only the default, copy and move
constructors of the types introduced in this extension can be used on host.

== Overview

This extension adds types that pack 4-bit integer values. That is, the 4-bit
integer types are represented by some subsection of a larger types inside the
storage of the packed type.


== Specification

=== Feature test macro

This extension provides a feature-test macro as described in the core SYCL
specification.  An implementation supporting this extension must predefine the
macro `SYCL_EXT_ONEAPI_PACKED_4BIT_INTEGER` to one of the values defined in the
table below.  Applications can test for the existence of this macro to determine
if the implementation supports this feature, or applications can test the
macro's value to determine which of the extension's features the implementation
supports.

[%header,cols="1,5"]
|===
|Value
|Description

|1
|The APIs of this experimental extension are not versioned, so the
 feature-test macro always has this value.
|===

=== New device aspect

This extension adds the `ext_oneapi_int4` enumerator to the `sycl::aspect`
enumeration.

```
namespace sycl {

enum class aspect : /*unspecified*/ {
  ext_oneapi_int4
};

} // namespace sycl
```

:optional-kernel-features: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:optional-kernel-features

When a device has the `ext_oneapi_int4` the `int4_packed` type can be used in
kernels on that device. If a device does not support this aspect, enqueuing a
kernel using this `int4_packed` will throw a synchronous `sycl::exception` with
`errc::kernel_not_supported`, as described in the
{optional-kernel-features}[optional kernel features] section of the SYCL 2020
specification.


=== New `int4_packed` type

The `int4_packed` type represents `NumElems` 4-bit signed integer values.  The
`StorageT` is the type used for packing the 4-bit values and must be either
`uint32_t` or `uint8_t`. If unspecified, `StorageT` will be `uint32_t` if
`NumElems` is divisible by 8 and `uint8_t` otherwise.

Aside from the default constructor, copy constructor and copy assignment
operator, none of the member functions of `int4_packed` can be called on
host. Calling any of these on host will cause a `sycl::exception` with
`errc::invalid` to be thrown.

==== Common conversion behavior [[int4_conv]]

The common conversion exhibits the following conversion behavior:

 * NaN, positive infinity and values greater than 7 are converted to 7.
 * Negative infinity and values less than -8 are converted to -8.
 * Other values are rounded towards zero.

==== `int4_packed` type interface

The `int4_packed` type is defined as:

[source]
----
namespace sycl::ext::oneapi::experimental {

template <size_t NumElems, typename StorageT = std::conditional_t<
                               NumElems % 8 == 0, uint32_t, uint8_t>>
class int4_packed {
public:
  int4_packed() = default;
  int4_packed(const int4_packed &) = default;
  int4_packed &operator=(const int4_packed &) = default;

  explicit int4_packed(const marray<half, NumElems> &vals);
  explicit int4_packed(const marray<bfloat16, NumElems> &vals);
  explicit int4_packed(const marray<int8_t, NumElems> &vals);
  explicit int4_packed(const marray<uint8_t, NumElems> &vals);

  int4_packed &operator=(const marray<half, NumElems> &vals);
  int4_packed &operator=(const marray<bfloat16, NumElems> &vals);
  int4_packed &operator=(const marray<int8_t, NumElems> &vals);
  int4_packed &operator=(const marray<uint8_t, NumElems> &vals);

  void assign(const marray<half, NumElems> &vals);
  void assign(const marray<bfloat16, NumElems> &vals);
  void assign(const marray<int8_t, NumElems> &vals);
  void assign(const marray<uint8_t, NumElems> &vals);

  template <typename TargetStorageT>
  operator int4_packed<NumElems, TargetStorageT>() const;

  operator marray<half, NumElems>() const;
  operator marray<bfloat16, NumElems>() const;
  operator marray<int8_t, NumElems>() const;
  operator marray<uint8_t, NumElems>() const;
};

} // namespace sycl::ext::oneapi::experimental
----

Table 5. Member functions of `int4_packed` class.
|===
| Member Function | Description

| `explicit int4_packed(const marray<half, NumElems> &vals)`
| Construct `int4_packed` by converting each value in `vals` to a value
  representable by a 4-bit signed integer.

  This conversion uses the <<int4_conv,common conversion behavior>>
  when converting each element of `vals` to a 4-bit signed integer value.

| `explicit int4_packed(const marray<bfloat16, NumElems> &vals)`
| Construct `int4_packed` by converting each value in `vals` to a value
  representable by a 4-bit signed integer.

  This conversion uses the <<int4_conv,common conversion behavior>>
  when converting each element of `vals` to a 4-bit signed integer value.

| `explicit int4_packed(const marray<int8_t, NumElems> &vals)`
| Construct `int4_packed` by converting each value in `vals` to a value
  representable by a 4-bit signed integer.

| `explicit int4_packed(const marray<uint8_t, NumElems> &vals)`
| Construct `int4_packed` by converting each value in `vals` to a value
  representable by a 4-bit signed integer.

| `int4_packed &operator=(const marray<half, NumElems> &vals)`
| Assigns the values in this instance of `int4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit signed
  integer.

  This conversion uses the <<int4_conv,common conversion behavior>>
  when converting each element of `vals` to a 4-bit signed integer value.

| `int4_packed &operator=(const marray<bfloat16, NumElems> &vals)`
| Assigns the values in this instance of `int4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit signed
  integer.

  This conversion uses the <<int4_conv,common conversion behavior>>
  when converting each element of `vals` to a 4-bit signed integer value.

| `int4_packed &operator=(const marray<int8_t, NumElems> &vals)`
| Assigns the values in this instance of `int4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit signed
  integer.

| `int4_packed &operator=(const marray<uint8_t, NumElems> &vals)`
| Assigns the values in this instance of `int4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit signed
  integer.

| `void assign(const marray<half, NumElems> &vals)`
| Assigns the values in this instance of `int4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit signed
  integer.

  This conversion uses the <<int4_conv,common conversion behavior>>
  when converting each element of `vals` to a 4-bit signed integer value.

| `void assign(const marray<bfloat16, NumElems> &vals)`
| Assigns the values in this instance of `int4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit signed
  integer.

  This conversion uses the <<int4_conv,common conversion behavior>>
  when converting each element of `vals` to a 4-bit signed integer value.

| `void assign(const marray<int8_t, NumElems> &vals)`
| Assigns the values in this instance of `int4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit signed
  integer.

| `void assign(const marray<uint8_t, NumElems> &vals)`
| Assigns the values in this instance of `int4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit signed
  integer.

| `template <typename TargetStorageT> operator int4_packed<NumElems, TargetStorageT>() const`
| Returns a `int4_packed<NumElems, TargetStorageT>` containing each 4-bit
  signed integer value in this instance of `int4_packed`

| `operator marray<half, NumElems>() const`
| Returns an `marray` containing each 4-bit signed integer value in this
  instance of `int4_packed` converted to a `half` value.

| `operator marray<bfloat16, NumElems>() const`
| Returns an `marray` containing each 4-bit signed integer value in this
  instance of `int4_packed` converted to a `bfloat16` value.

| `operator marray<int8_t, NumElems>() const`
| Returns an `marray` containing each 4-bit signed integer value in this
  instance of `int4_packed` converted to a `int8_t` value.

| `operator marray<uint8_t, NumElems>() const`
| Returns an `marray` containing each 4-bit signed integer value in this
  instance of `int4_packed` converted to a `uint8_t` value.

|===


=== New `uint4_packed` type

The `uint4_packed` type represents `NumElems` 4-bit unsigned integer values.
The `StorageT` is the type used for packing the 4-bit values and must be either
`uint32_t` or `uint8_t`. If unspecified, `StorageT` will be `uint32_t` if
`NumElems` is divisible by 8 and `uint8_t` otherwise.

Aside from the default constructor, copy constructor and copy assignment
operator, none of the member functions of `uint4_packed` can be called on
host. Calling any of these on host will cause a `sycl::exception` with
`errc::invalid` to be thrown.

==== Common conversion behavior [[uint4_conv]]

Conversions exhibits the following behavior:

 * NaN, positive infinity and values greater than 15 are converted to 15.
 * Negative infinity and values less than 0 are converted to 0.
 * Other values are rounded to zero.

==== `uint4_packed` type interface

The `uint4_packed` type is defined as:

[source]
----
namespace sycl::ext::oneapi::experimental {

template <size_t NumElems, typename StorageT = std::conditional_t<
                               NumElems % 8 == 0, uint32_t, uint8_t>>
class uint4_packed {
public:
  uint4_packed() = default;
  uint4_packed(const uint4_packed &) = default;
  uint4_packed &operator=(const uint4_packed &) = default;

  explicit uint4_packed(const marray<half, NumElems> &vals);
  explicit uint4_packed(const marray<bfloat16, NumElems> &vals);
  explicit uint4_packed(const marray<int8_t, NumElems> &vals);
  explicit uint4_packed(const marray<uint8_t, NumElems> &vals);

  uint4_packed &operator=(const marray<half, NumElems> &vals);
  uint4_packed &operator=(const marray<bfloat16, NumElems> &vals);
  uint4_packed &operator=(const marray<int8_t, NumElems> &vals);
  uint4_packed &operator=(const marray<uint8_t, NumElems> &vals);

  void assign(const marray<half, NumElems> &vals);
  void assign(const marray<bfloat16, NumElems> &vals);
  void assign(const marray<int8_t, NumElems> &vals);
  void assign(const marray<uint8_t, NumElems> &vals);

  template <typename TargetStorageT>
  operator uint4_packed<NumElems, TargetStorageT>() const;

  operator marray<half, NumElems>() const;
  operator marray<bfloat16, NumElems>() const;
  operator marray<int8_t, NumElems>() const;
  operator marray<uint8_t, NumElems>() const;
};

} // namespace sycl::ext::oneapi::experimental
----

Table 5. Member functions of `uint4_packed` class.
|===
| Member Function | Description

| `explicit uint4_packed(const marray<half, NumElems> &vals)`
| Construct `uint4_packed` by converting each value in `vals` to a value
  representable by a 4-bit unsigned integer.

  This conversion uses the <<uint4_conv,common conversion behavior>>
  when converting each element of `vals` to a 4-bit unsigned integer value.

| `explicit uint4_packed(const marray<bfloat16, NumElems> &vals)`
| Construct `uint4_packed` by converting each value in `vals` to a value
  representable by a 4-bit unsigned integer.

  This conversion uses the <<uint4_conv,common conversion behavior>>
  when converting each element of `vals` to a 4-bit unsigned integer value.

| `explicit uint4_packed(const marray<int8_t, NumElems> &vals)`
| Construct `uint4_packed` by converting each value in `vals` to a value
  representable by a 4-bit unsigned integer.

| `explicit uint4_packed(const marray<uint8_t, NumElems> &vals)`
| Construct `uint4_packed` by converting each value in `vals` to a value
  representable by a 4-bit unsigned integer.

| `uint4_packed &operator=(const marray<half, NumElems> &vals)`
| Assigns the values in this instance of `uint4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit unsigned
  integer.

  This conversion uses the <<uint4_conv,common conversion behavior>>
  when converting each element of `vals` to a 4-bit unsigned integer value.

| `uint4_packed &operator=(const marray<bfloat16, NumElems> &vals)`
| Assigns the values in this instance of `uint4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit unsigned
  integer.

  This conversion uses the <<uint4_conv,common conversion behavior>>
  when converting each element of `vals` to a 4-bit unsigned integer value.

| `uint4_packed &operator=(const marray<int8_t, NumElems> &vals)`
| Assigns the values in this instance of `uint4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit unsigned
  integer.

| `uint4_packed &operator=(const marray<uint8_t, NumElems> &vals)`
| Assigns the values in this instance of `uint4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit unsigned
  integer.

| `void assign(const marray<half, NumElems> &vals)`
| Assigns the values in this instance of `uint4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit unsigned
  unsigned integer.

  This conversion uses the <<uint4_conv,common conversion behavior>>
  when converting each element of `vals` to a 4-bit unsigned integer value.

| `void assign(const marray<bfloat16, NumElems> &vals)`
| Assigns the values in this instance of `uint4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit unsigned
  integer.

  This conversion uses the <<uint4_conv,common conversion behavior>>
  when converting each element of `vals` to a 4-bit unsigned integer value.

| `void assign(const marray<int8_t, NumElems> &vals)`
| Assigns the values in this instance of `uint4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit unsigned
  integer.

| `void assign(const marray<uint8_t, NumElems> &vals)`
| Assigns the values in this instance of `uint4_packed` to the values in
  `vals` after converting them to values representable by a 4-bit unsigned
  integer.

| `template <typename TargetStorageT> operator uint4_packed<NumElems, TargetStorageT>() const`
| Returns a `uint4_packed<NumElems, TargetStorageT>` containing each 4-bit
  unsigned integer value in this instance of `uint4_packed`.

| `operator marray<half, NumElems>() const`
| Returns an `marray` containing each 4-bit unsigned integer value in this
  instance of `uint4_packed` converted to a `half` value.

| `operator marray<bfloat16, NumElems>() const`
| Returns an `marray` containing each 4-bit unsigned integer value in this
  instance of `uint4_packed` converted to a `bfloat16` value.

| `operator marray<int8_t, NumElems>() const`
| Returns an `marray` containing each 4-bit unsigned integer value in this
  instance of `uint4_packed` converted to a `int8_t` value.

| `operator marray<uint8_t, NumElems>() const`
| Returns an `marray` containing each 4-bit unsigned integer value in this
  instance of `uint4_packed` converted to a `uint8_t` value.

|===

