//==- variadic_parallel_for_utils.hpp - SYCL variable utilities --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
// Utility functions for parallel_for with variadic parameters, such as reducers
// and item views.

#pragma once

#include <CL/sycl/detail/cg.hpp>
#include <CL/sycl/detail/tuple.hpp>
#include <CL/sycl/kernel.hpp>

#include <tuple>
#include <type_traits>
#include <utility>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

namespace ext {
namespace codeplay {
namespace detail {
// Forward declaration.
template <typename T> struct IsItemView;
} // namespace detail
} // namespace codeplay

namespace oneapi {
namespace detail {

// Forward declaration.
template <typename T> struct IsReduction;

/// These are the forward declaration for the classes that help to create
/// names for additional kernels. It is used only when there are
/// more then 1 kernels in one parallel_for() implementing SYCL reduction.
template <typename T1, bool B1, bool B2, typename T2>
class __sycl_reduction_main_kernel;
template <typename T1, bool B1, bool B2, typename T2>
class __sycl_reduction_aux_kernel;

/// Helper structs to get additional kernel name types based on given
/// \c Name and additional template parameters helping to distinguish kernels.
/// If \c Name is undefined (is \c auto_name) leave it that way to take
/// advantage of unnamed kernels being named after their functor.
template <typename Name, typename Type, bool B1, bool B2, typename T3 = void>
struct get_reduction_main_kernel_name_t {
  using name = __sycl_reduction_main_kernel<Name, B1, B2, T3>;
};
template <typename Type, bool B1, bool B2, typename T3>
struct get_reduction_main_kernel_name_t<sycl::detail::auto_name, Type, B1, B2,
                                        T3> {
  using name = sycl::detail::auto_name;
};
template <typename Name, typename Type, bool B1, bool B2, typename T3>
struct get_reduction_aux_kernel_name_t {
  using name = __sycl_reduction_aux_kernel<Name, B1, B2, T3>;
};
template <typename Type, bool B1, bool B2, typename T3>
struct get_reduction_aux_kernel_name_t<sycl::detail::auto_name, Type, B1, B2,
                                       T3> {
  using name = sycl::detail::auto_name;
};

/// Called in device code. This function iterates through the index space
/// \p Range using stride equal to the global range specified in \p NdId,
/// which gives much better performance than using stride equal to 1.
/// For each of the index the given \p F function/functor is called and
/// the reduction value hold in \p Reducer is accumulated in those calls.
template <typename KernelFunc, int Dims, typename ReducerT>
void reductionLoop(const range<Dims> &Range, ReducerT &Reducer,
                   const nd_item<1> &NdId, KernelFunc &F) {
  size_t Start = NdId.get_global_id(0);
  size_t End = Range.size();
  size_t Stride = NdId.get_global_range(0);
  for (size_t I = Start; I < End; I += Stride)
    F(sycl::detail::getDelinearizedId(Range, I), Reducer);
}

/// Predicate returning true if all template type parameter is a item view or
/// a reduction.
template <typename T> struct IsValidParam {
  static constexpr bool value =
      IsReduction<T>::value ||
      cl::sycl::ext::codeplay::detail::IsItemView<T>::value;
};

/// Predicate returning true if all template type parameters except the last one
/// are reduction or a item view.
template <typename FirstT, typename... RestT> struct AreAllButLastValidParam {
  static constexpr bool value =
      IsValidParam<FirstT>::value && AreAllButLastValidParam<RestT...>::value;
};

/// Helper specialization of AreAllButLastValidParam for one element only.
/// Returns true if the template parameter is not a reduction or a item view.
template <typename T> struct AreAllButLastValidParam<T> {
  static constexpr bool value = !IsValidParam<T>::value;
};

template <typename FirstT, typename... RestT> struct ReductionParamCount {
  static constexpr size_t value =
      ReductionParamCount<FirstT>::value + ReductionParamCount<RestT...>::value;
};

template <typename T> struct ReductionParamCount<T> {
  static constexpr size_t value = IsReduction<T>::value ? 1 : 0;
};

// Converts a tuple of integrals to an index sequence.
template <typename Tuple> struct tuple_to_index_sequence;

template <typename... Ts> struct tuple_to_index_sequence<std::tuple<Ts...>> {
  using type = std::index_sequence<Ts::value...>;
};

template <template <typename> class Pred, typename Tuple, typename Seq>
struct make_filtered_index_sequence;

template <template <typename> class Pred, typename Tuple, std::size_t... Is>
struct make_filtered_index_sequence<Pred, Tuple, std::index_sequence<Is...>> {
  using type = typename tuple_to_index_sequence<decltype(std::tuple_cat(
      std::conditional_t<Pred<std::tuple_element_t<Is, Tuple>>::value,
                         std::tuple<std::integral_constant<std::size_t, Is>>,
                         std::tuple<>>{}...))>::type;
};

template <typename Tuple, typename Seq>
using make_reduction_index_sequence =
    make_filtered_index_sequence<IsReduction, Tuple, Seq>;

template <typename Tuple, typename Seq>
using make_item_view_index_sequence =
    make_filtered_index_sequence<cl::sycl::ext::codeplay::detail::IsItemView,
                                 Tuple, Seq>;

// std::tuple seems to be a) too heavy and b) not copyable to device now
// Thus sycl::detail::tuple is used instead.
// Switching from sycl::device::tuple to std::tuple can be done by re-defining
// the ReduTupleT type and makeReduTupleT() function below.
template <typename... Ts> using ReduTupleT = sycl::detail::tuple<Ts...>;
template <typename... Ts> ReduTupleT<Ts...> makeReduTupleT(Ts... Elements) {
  return sycl::detail::make_tuple(Elements...);
}

/// For the given 'Reductions' types pack and indices enumerating only
/// the reductions for which a local accessors are needed, this function creates
/// those local accessors and returns a tuple consisting of them.
template <typename... Reductions, size_t... Is>
auto createReduLocalAccs(size_t Size, handler &CGH,
                         std::index_sequence<Is...>) {
  return makeReduTupleT(
      std::tuple_element_t<Is, std::tuple<Reductions...>>::getReadWriteLocalAcc(
          Size, CGH)...);
}

/// For the given 'Reductions' types pack and indices enumerating them this
/// function either creates new temporary accessors for partial sums (if IsOneWG
/// is false) or returns user's accessor/USM-pointer if (IsOneWG is true).
template <bool IsOneWG, typename... Reductions, size_t... Is>
auto createReduOutAccs(size_t NWorkGroups, handler &CGH,
                       std::tuple<Reductions...> &ReduTuple,
                       std::index_sequence<Is...>) {
  return makeReduTupleT(
      std::get<Is>(ReduTuple).template getWriteMemForPartialReds<IsOneWG>(
          NWorkGroups, CGH)...);
}

/// For the given 'Reductions' types pack and indices enumerating them this
/// function returns accessors to buffers holding partial sums generated in the
/// previous kernel invocation.
template <typename... Reductions, size_t... Is>
auto getReadAccsToPreviousPartialReds(handler &CGH,
                                      std::tuple<Reductions...> &ReduTuple,
                                      std::index_sequence<Is...>) {
  return makeReduTupleT(
      std::get<Is>(ReduTuple).getReadAccToPreviousPartialReds(CGH)...);
}

template <typename... Reductions, size_t... Is>
ReduTupleT<typename Reductions::result_type...>
getReduIdentities(std::tuple<Reductions...> &ReduTuple,
                  std::index_sequence<Is...>) {
  return {std::get<Is>(ReduTuple).getIdentity()...};
}

template <typename... Reductions, size_t... Is>
ReduTupleT<typename Reductions::binary_operation...>
getReduBOPs(std::tuple<Reductions...> &ReduTuple, std::index_sequence<Is...>) {
  return {std::get<Is>(ReduTuple).getBinaryOperation()...};
}

template <typename... Reductions, size_t... Is>
std::array<bool, sizeof...(Reductions)>
getInitToIdentityProperties(std::tuple<Reductions...> &ReduTuple,
                            std::index_sequence<Is...>) {
  return {std::get<Is>(ReduTuple).initializeToIdentity()...};
}

template <typename... Reductions, size_t... Is>
std::tuple<typename Reductions::reducer_type...>
createReducers(ReduTupleT<typename Reductions::result_type...> Identities,
               ReduTupleT<typename Reductions::binary_operation...> BOPsTuple,
               std::index_sequence<Is...>) {
  return {typename Reductions::reducer_type{std::get<Is>(Identities),
                                            std::get<Is>(BOPsTuple)}...};
}

template <typename KernelType, typename... Args> struct checkKernelSignature {
  template <typename K>
  static auto check(K *k)
      -> decltype((*k)(std::declval<Args>()...), void(), std::true_type());
  template <typename K> static auto check(...) -> decltype(std::false_type());
  static constexpr bool value = decltype(check<KernelType>(0))::value;
};

template <typename KernelType, int Dims, typename... ArgT, size_t... Is>
void callUserKernelFunc(KernelType KernelFunc, nd_item<Dims> NDIt,
                        std::tuple<ArgT...> &Args, std::index_sequence<Is...>) {
  if constexpr (checkKernelSignature<KernelType, nd_item<Dims>,
                                     decltype(std::get<Is>(Args))...>::value) {
    KernelFunc(NDIt, std::get<Is>(Args)...);
  } else {
    static_assert(checkKernelSignature<KernelType, decltype(std::get<Is>(
                                                       Args))...>::value &&
                  "Kernel function does not have suitable parameters.");
    KernelFunc(std::get<Is>(Args)...);
  }
}

template <bool Pow2WG, typename... LocalAccT, typename... ReducerT,
          typename... ResultT, size_t... Is>
void initReduLocalAccs(size_t LID, size_t WGSize,
                       ReduTupleT<LocalAccT...> LocalAccs,
                       const std::tuple<ReducerT...> &Reducers,
                       ReduTupleT<ResultT...> Identities,
                       std::index_sequence<Is...>) {
  std::tie(std::get<Is>(LocalAccs)[LID]...) =
      std::make_tuple(std::get<Is>(Reducers).MValue...);

  // For work-groups, which size is not power of two, local accessors have
  // an additional element with index WGSize that is used by the tree-reduction
  // algorithm. Initialize those additional elements with identity values here.
  if (!Pow2WG)
    std::tie(std::get<Is>(LocalAccs)[WGSize]...) =
        std::make_tuple(std::get<Is>(Identities)...);
}

template <bool UniformPow2WG, typename... LocalAccT, typename... InputAccT,
          typename... ResultT, size_t... Is>
void initReduLocalAccs(size_t LID, size_t GID, size_t NWorkItems, size_t WGSize,
                       ReduTupleT<InputAccT...> LocalAccs,
                       ReduTupleT<LocalAccT...> InputAccs,
                       ReduTupleT<ResultT...> Identities,
                       std::index_sequence<Is...>) {
  // Normally, the local accessors are initialized with elements from the input
  // accessors. The exception is the case when (GID >= NWorkItems), which
  // possible only when UniformPow2WG is false. For that case the elements of
  // local accessors are initialized with identity value, so they would not
  // give any impact into the final partial sums during the tree-reduction
  // algorithm work.
  if (UniformPow2WG || GID < NWorkItems)
    std::tie(std::get<Is>(LocalAccs)[LID]...) =
        std::make_tuple(std::get<Is>(InputAccs)[GID]...);
  else
    std::tie(std::get<Is>(LocalAccs)[LID]...) =
        std::make_tuple(std::get<Is>(Identities)...);

  // For work-groups, which size is not power of two, local accessors have
  // an additional element with index WGSize that is used by the tree-reduction
  // algorithm. Initialize those additional elements with identity values here.
  if (!UniformPow2WG)
    std::tie(std::get<Is>(LocalAccs)[WGSize]...) =
        std::make_tuple(std::get<Is>(Identities)...);
}

template <typename... LocalAccT, typename... BOPsT, size_t... Is>
void reduceReduLocalAccs(size_t IndexA, size_t IndexB,
                         ReduTupleT<LocalAccT...> LocalAccs,
                         ReduTupleT<BOPsT...> BOPs,
                         std::index_sequence<Is...>) {
  std::tie(std::get<Is>(LocalAccs)[IndexA]...) =
      std::make_tuple((std::get<Is>(BOPs)(std::get<Is>(LocalAccs)[IndexA],
                                          std::get<Is>(LocalAccs)[IndexB]))...);
}

template <bool Pow2WG, bool IsOneWG, typename... Reductions,
          typename... OutAccT, typename... LocalAccT, typename... BOPsT,
          typename... Ts, size_t... Is>
void writeReduSumsToOutAccs(
    size_t OutAccIndex, size_t WGSize, std::tuple<Reductions...> *,
    ReduTupleT<OutAccT...> OutAccs, ReduTupleT<LocalAccT...> LocalAccs,
    ReduTupleT<BOPsT...> BOPs, ReduTupleT<Ts...> IdentityVals,
    std::array<bool, sizeof...(Reductions)> IsInitializeToIdentity,
    std::index_sequence<Is...>) {
  // Add the initial value of user's variable to the final result.
  if (IsOneWG)
    std::tie(std::get<Is>(LocalAccs)[0]...) = std::make_tuple(std::get<Is>(
        BOPs)(std::get<Is>(LocalAccs)[0],
              IsInitializeToIdentity[Is]
                  ? std::get<Is>(IdentityVals)
                  : std::tuple_element_t<Is, std::tuple<Reductions...>>::
                        getOutPointer(std::get<Is>(OutAccs))[0])...);

  if (Pow2WG) {
    // The partial sums for the work-group are stored in 0-th elements of local
    // accessors. Simply write those sums to output accessors.
    std::tie(std::tuple_element_t<Is, std::tuple<Reductions...>>::getOutPointer(
        std::get<Is>(OutAccs))[OutAccIndex]...) =
        std::make_tuple(std::get<Is>(LocalAccs)[0]...);
  } else {
    // Each of local accessors keeps two partial sums: in 0-th and WGsize-th
    // elements. Combine them into final partial sums and write to output
    // accessors.
    std::tie(std::tuple_element_t<Is, std::tuple<Reductions...>>::getOutPointer(
        std::get<Is>(OutAccs))[OutAccIndex]...) =
        std::make_tuple(std::get<Is>(BOPs)(std::get<Is>(LocalAccs)[0],
                                           std::get<Is>(LocalAccs)[WGSize])...);
  }
}

// Concatenate an empty sequence.
constexpr std::index_sequence<> concat_sequences(std::index_sequence<>) {
  return {};
}

// Concatenate a sequence consisting of 1 element.
template <size_t I>
constexpr std::index_sequence<I> concat_sequences(std::index_sequence<I>) {
  return {};
}

// Concatenate two potentially empty sequences.
template <size_t... Is, size_t... Js>
constexpr std::index_sequence<Is..., Js...>
concat_sequences(std::index_sequence<Is...>, std::index_sequence<Js...>) {
  return {};
}

// Concatenate more than 2 sequences.
template <size_t... Is, size_t... Js, class... Rs>
constexpr auto concat_sequences(std::index_sequence<Is...>,
                                std::index_sequence<Js...>, Rs...) {
  return concat_sequences(std::index_sequence<Is..., Js...>{}, Rs{}...);
}

struct IsRWReductionPredicate {
  template <typename T> struct Func {
    static constexpr bool value =
        std::remove_pointer_t<T>::accessor_mode == access::mode::read_write;
  };
};

struct IsNonUsmReductionPredicate {
  template <typename T> struct Func {
    static constexpr bool value = !std::remove_pointer_t<T>::is_usm;
  };
};

struct EmptyReductionPredicate {
  template <typename T> struct Func { static constexpr bool value = false; };
};

struct IsReadableItemViewPredicate {
  template <typename T> struct Func {
    static constexpr bool value =
        std::remove_pointer_t<T>::accessor_mode != access::mode::write &&
        std::remove_pointer_t<T>::accessor_mode != access::mode::discard_write;
  };
};

struct IsWriteableItemViewPredicate {
  template <typename T> struct Func {
    static constexpr bool value =
        std::remove_pointer_t<T>::accessor_mode != access::mode::read;
  };
};

template <bool Cond, size_t I> struct FilterElement {
  using type =
      std::conditional_t<Cond, std::index_sequence<I>, std::index_sequence<>>;
};

/// For each index 'I' from the given indices pack 'Is' this function initially
/// creates a number of short index_sequences, where each of such short
/// index sequences is either empty (if the given Functor returns false for the
/// type T[I]) or 1 element 'I' (otherwise). After that this function
/// concatenates those short sequences into one and returns the result sequence.
template <typename... T, typename FunctorT, size_t... Is,
          std::enable_if_t<(sizeof...(Is) > 0), int> Z = 0>
constexpr auto filterSequenceHelper(FunctorT, std::index_sequence<Is...>) {
  return concat_sequences(
      typename FilterElement<FunctorT::template Func<std::tuple_element_t<
                                 Is, std::tuple<T...>>>::value,
                             Is>::type{}...);
}
template <typename... T, typename FunctorT, size_t... Is,
          std::enable_if_t<(sizeof...(Is) == 0), int> Z = 0>
constexpr auto filterSequenceHelper(FunctorT, std::index_sequence<Is...>) {
  return std::index_sequence<>{};
}

/// For each index 'I' from the given indices pack 'Is' this function returns
/// an index sequence consisting of only those 'I's for which the 'FunctorT'
/// applied to 'T[I]' returns true.
template <typename... T, typename FunctorT, size_t... Is>
constexpr auto filterSequence(FunctorT F, std::index_sequence<Is...> Indices) {
  return filterSequenceHelper<T...>(F, Indices);
}

constexpr size_t reduGetMemPerWorkItemHelper() { return 0; }

template <typename Reduction> size_t reduGetMemPerWorkItemHelper(Reduction &) {
  return sizeof(typename Reduction::result_type);
}

template <typename Reduction, typename... RestT>
size_t reduGetMemPerWorkItemHelper(Reduction &, RestT... Rest) {
  return sizeof(typename Reduction::result_type) +
         reduGetMemPerWorkItemHelper(Rest...);
}

template <typename... ReductionT, size_t... Is>
size_t reduGetMemPerWorkItem(std::tuple<ReductionT...> &ReduTuple,
                             std::index_sequence<Is...>) {
  return reduGetMemPerWorkItemHelper(std::get<Is>(ReduTuple)...);
}

/// Utility function: for the given tuple \param Tuple the function returns
/// a new tuple consisting of only elements indexed by the index sequence.
template <typename TupleT, std::size_t... Is>
std::tuple<std::tuple_element_t<Is, TupleT>...>
tuple_select_elements(TupleT Tuple, std::index_sequence<Is...>) {
  return {std::get<Is>(std::move(Tuple))...};
}

template <int Dims, typename ReadToTupleT, typename ItemViewAccessorTupleT>
void readItemViewValueToTuple(id<Dims>, ReadToTupleT &,
                              ItemViewAccessorTupleT &, std::index_sequence<>) {
}

template <int Dims, typename ReadToTupleT, typename ItemViewAccessorTupleT,
          size_t I, size_t... Is>
void readItemViewValueToTuple(id<Dims> gid, ReadToTupleT &vals,
                              ItemViewAccessorTupleT &streamAccs,
                              std::index_sequence<I, Is...>) {
  std::get<I>(vals) = std::get<I>(streamAccs)[gid];
  readItemViewValueToTuple(gid, vals, streamAccs, std::index_sequence<Is...>{});
}

template <int Dims, typename ItemViewAccessorTupleT, typename ReadToTupleT>
void writeTupleValueToItemView(id<Dims>, ReadToTupleT &,
                               ItemViewAccessorTupleT &,
                               std::index_sequence<>) {}

template <int Dims, typename ItemViewAccessorTupleT, typename ReadToTupleT,
          size_t I, size_t... Is>
void writeTupleValueToItemView(id<Dims> gid, ItemViewAccessorTupleT &streamAccs,
                               ReadToTupleT &vals,
                               std::index_sequence<I, Is...>) {
  std::get<I>(streamAccs)[gid] = std::get<I>(vals);
  writeTupleValueToItemView(gid, streamAccs, vals,
                            std::index_sequence<Is...>{});
}

template <typename ItemViewValueTupleT, typename... ItemViews, size_t... Is>
constexpr std::tuple<typename ItemViews::reference...>
getReferenceTuple(ItemViewValueTupleT &vals, std::tuple<ItemViews...>,
                  std::index_sequence<Is...>) {
  return {const_cast<typename ItemViews::reference>(std::get<Is>(vals))...};
}

} // namespace detail
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
