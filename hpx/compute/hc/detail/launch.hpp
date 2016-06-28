///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_HC_DETAIL_LAUNCH_HPP
#define HPX_COMPUTE_HC_DETAIL_LAUNCH_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_HC)
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke_fused.hpp>

#include <hpx/compute/hc/config.hpp>
#include <hpx/compute/hc/target.hpp>

namespace hpx { namespace compute { namespace hc { namespace detail
{

    // Launch any given function F with the given parameters. This function
    // does not involve any device synchronization.
    template <typename F, typename ...Args>
    void launch(target const& t, int tile_count, int threads_per_tile, F && f,
        Args&&... args)
    {
#if !defined(__COMPUTE__ACCELERATOR__)
        hc::parallel_for_each(
            hc::extent<1>(threads_to_launch),
            [=](index<1> idx) [[hc]]
            {
                f(idx, args);
            }
        );
#endif
    }

    // Launch any given function F with the given parameters. This function
    // does not involve any device synchronization.
    template <typename F, typename ...Args>
    HPX_HOST_DEVICE
    void launch(target const& t, int threads_to_launch, F && f,
        Args const &... args)
    {
#if !defined(__COMPUTE__ACCELERATOR__)
        hc::parallel_for_each(
            hc::extent<1>(threads_to_launch),
            [=](index<1> idx) [[hc]]
            {
                f(idx, args);
            }
        );
#endif
    }
}}}}

#endif
#endif
