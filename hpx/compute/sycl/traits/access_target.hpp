///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_SYCL_TARGET_TRAITS_HPP
#define HPX_COMPUTE_SYCL_TARGET_TRAITS_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SYCL)

#include <hpx/compute/traits/access_target.hpp>
#include <hpx/compute/sycl/target.hpp>


namespace hpx { namespace compute { namespace traits
{
    template <>
    struct access_target<sycl::target>
    {
        typedef sycl::target target_type;
        template <typename T>
        static T read(sycl::buffer_t<T> * buffer, uint64_t pos)
        {
            auto host_access = buffer->template get_access<
                    cl::sycl::access::mode::read,
                    cl::sycl::access::target::host_buffer
                >();
            return host_access.get_pointer()[pos];
        }

        template <typename T>
        static void write(sycl::buffer_t<T> * buffer, uint64_t pos, const T & val)
        {
            auto host_access = buffer->template get_access<
                    cl::sycl::access::mode::read,
                    cl::sycl::access::target::host_buffer
                >();
            host_access.get_pointer()[pos] = val;
        }
    };
}}}

#endif
#endif
