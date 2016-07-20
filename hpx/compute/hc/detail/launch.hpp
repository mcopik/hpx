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
#include <hpx/compute/hc/target_ptr.hpp>

namespace hpx { namespace compute { namespace hc { namespace detail
{

    namespace detail
    {
        template<typename T>
        using kernel_transfer_type =
            std::tuple<buffer_acc_t<T>, std::ptrdiff_t>;//typename target_ptr<T>::difference_type>;

        template<typename T>
        const T & extract_target_ptr(const T & t) [[hc, cpu]]
        {
            return t;
        }

        template<typename T>
        kernel_transfer_type<int>//typename T::proxy_type::value_type>
        extract_target_ptr(const target_ptr<T> & t) [[hc, cpu]]
        {
            return std::make_tuple(t.device_ptr()->get_buffer_acc(), t.pos());
        }

        template<typename T>
        const T & make_target_ptr(const T & t) [[hc, cpu]]
        {
            return t;
        }

        template<typename T>
        target_ptr< buffer_proxy<T> > make_target_ptr(const kernel_transfer_type<T> & t) [[hc, cpu]]
        {
            // Recreate the target_ptr
            buffer_proxy<T> proxy(std::get<0>(t));
            // Move the buffer proxy and pass position
            return target_ptr< buffer_proxy<T> >(std::move(proxy), nullptr,
                    std::get<1>(t));
        }

        template <typename F, typename ...Args>
        void launch(target const& t, int tile_count, int threads_per_tile, F && f,
            const Args &... args)
        {
    //#if !defined(__COMPUTE__ACCELERATOR__)
            ::hc::parallel_for_each(t.native_handle().get_device(),
                global_size<1>(tile_count*threads_per_tile).tile(threads_per_tile),
                [=](local_index<1> idx) mutable [[hc]]
                {
                    f(idx, args...);//make_target_ptr(args)...);
                }
            );
    //#else
            //
    //#endif
        }

        // Launch any given function F with the given parameters. This function
        // does not involve any device synchronization.
        template <typename F, typename ...Args>
        HPX_HOST_DEVICE
        void launch(target const& t, int threads_to_launch, F && f,
            Args const &... args)
        {
    //#if !defined(__COMPUTE__ACCELERATOR__)
            ::hc::parallel_for_each(
                global_size<1>(threads_to_launch),
                [=](index<1> idx) mutable [[hc]]
                {
                    f(idx, args...);//make_target_ptr(args)...);
                }
            );

        }
    }

    // Launch any given function F with the given parameters. This function
    // does not involve any device synchronization.
    template <typename F, typename ...Args>
    void launch(target const& t, int tile_count, int threads_per_tile, F && f,
        Args&&... args)
    {
        detail::launch(t, tile_count, threads_per_tile, std::forward<F>(f),
                std::forward<Args&&>(args)...);//detail::extract_target_ptr(args)...);
    }

    // Launch any given function F with the given parameters. This function
    // does not involve any device synchronization.
    template <typename F, typename ...Args>
    HPX_HOST_DEVICE
    void launch(target const& t, int threads_to_launch, F && f,
        Args... args)//Args const &... args)
    {
        detail::launch(t, threads_to_launch, std::forward<F>(f),
                std::forward<Args&&>(args)...);//detail::extract_target_ptr(args)...);
    }

}}}}

#endif
#endif
