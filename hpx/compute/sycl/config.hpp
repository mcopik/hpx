//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_SYCL_CONFIG_HPP
#define HPX_COMPUTE_SYCL_CONFIG_HPP

#if defined(HPX_HAVE_SYCL)

#include <SYCL/sycl.hpp>


namespace hpx { namespace compute { namespace sycl {
    typedef ::cl::sycl::device device_t;
    typedef ::cl::sycl::queue queue_t;
    typedef ::cl::sycl::exception exception_t;
}}}

#endif
#endif
