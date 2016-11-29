//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_SYCL_SERIALIZATION_VALUE_PROXY_HPP
#define HPX_COMPUTE_SYCL_SERIALIZATION_VALUE_PROXY_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SYCL)

#include <hpx/compute/sycl/value_proxy.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

namespace hpx { namespace serialization
{
    template <typename T>
    void serialize(input_archive & ar, compute::sycl::value_proxy<T> & v,
        unsigned)
    {
        T t;
        ar >> t;
        v = t;
    }

    template <typename T>
    void serialize(output_archive & ar, compute::sycl::value_proxy<T> const& v,
        unsigned)
    {
        ar << T(v);
    }
}}

#endif
#endif
