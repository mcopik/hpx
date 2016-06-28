//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_HC_CONFIG_HPP
#define HPX_COMPUTE_HC_CONFIG_HPP

#if defined(HPX_HAVE_HC)

#include <hc.hpp>

#define __COMPUTE__ACCELERATOR__ __KALMAR_ACCELERATOR__

namespace hpx { namespace compute { namespace hc {
    template<size_t N>
    using index = ::hc::index<N>;

    template<size_t N>
    using global_size = ::hc::extent<N>;

    template<size_t N>
    using local_index = ::hc::tiled_index<N>;

    template<size_t N>
    using local_size = ::hc::tiled_extent<N>;

    template<typename T>
    using buffer_t = ::hc::array<T>;

    template<typename T>
    using buffer_acc_t = ::hc::array_view<T>;

    typedef ::hc::accelerator_view device_t;
    typedef ::hc::runtime_exception exception_t;
}}}

#endif

#endif
