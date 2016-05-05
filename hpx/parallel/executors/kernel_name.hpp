//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/default_chunk_size.hpp

#if !defined(HPX_PARALLEL_DEFAULT_CHUNK_SIZE)
#define HPX_PARALLEL_DEFAULT_CHUNK_SIZE

#include <hpx/config.hpp>
#include <hpx/traits/is_executor_parameters.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_parameter_traits.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    /// If \a chunk size is not specified by the user, the chosen size will be
    /// constant and defined by the template parameter.
	///
    /// \note This executor parameters type should be used to mark a constant
	/// behaviour of an executor which doesn't depend on the hardware.
    ///
	template<typename KernelName>
    struct kernel_name : executor_parameters_tag
    {
        typedef KernelName name;
    };
}}}

#endif
