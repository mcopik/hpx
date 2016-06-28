///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Marcin COpik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_HC_TARGET_TRAITS_HPP
#define HPX_COMPUTE_HC_TARGET_TRAITS_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_HC)
#include <hpx/compute/hc/config.hpp>
#include <hpx/compute/traits/access_target.hpp>
#include <hpx/compute/hc/target.hpp>


namespace hpx { namespace compute { namespace traits
{
    template <>
    struct access_target<hc::target>
    {
        typedef hc::target target_type;

        template <typename T>
        static T read(hc::target const& tgt, T const* t)
        {
#if defined(__COMPUTE__ACCELERATOR__)
            return *t;
#else
            T tmp;
            //cudaMemcpyAsync(&tmp, t, sizeof(T), cudaMemcpyDeviceToHost,
            //    tgt.native_handle().get_stream());
            //tgt.synchronize();
            hc::copy_async()
            std::cout << "READ_TRAITS" << '\n';
            return tmp;
#endif
        }

        template <typename T>
        static void write(hc::target const& tgt, T* dst, T const* src)
        {
#if defined(__COMPUTE__ACCELERATOR__)
            *dst = *src;
#else
//            cudaMemcpyAsync(dst, src, sizeof(T), cudaMemcpyHostToDevice,
//                tgt.native_handle().get_stream());
            std::cout << "WRITE_TRAITS" << '\n';
#endif
        }
    };
}}}

#endif
#endif
