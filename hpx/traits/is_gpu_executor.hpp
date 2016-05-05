//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_GPU_EXECUTOR)
#define HPX_TRAITS_IS_GPU_EXECUTOR

#include <hpx/traits.hpp>
#include <hpx/config/inline_namespace.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>

#include <boost/type_traits/is_base_of.hpp>

// Useful only if a compiler supporting GPU code generation is used
#if defined(HPX_WITH_GPU_EXECUTOR)

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    struct gpu_executor_tag {};

    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_gpu_executor
          : boost::is_base_of<gpu_executor_tag, T>
        {};

        template <>
        struct is_gpu_executor<executor_tag>
          : std::false_type
        {};
        /// \endcond
    }

    template <typename T>
    struct is_gpu_executor
      : detail::is_gpu_executor<typename hpx::util::decay<T>::type>
    {};
}}}

namespace hpx { namespace traits
{
    // new executor framework
    template <typename Executor>
	//TODO: why it doesn't work?, typename Enable>
    struct is_gpu_executor
      : parallel::v3::is_gpu_executor<Executor>
    {};
}}

#endif

#endif

