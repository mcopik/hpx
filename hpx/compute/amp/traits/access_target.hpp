///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Marcin COpik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_AMP_TARGET_TRAITS_HPP
#define HPX_COMPUTE_AMP_TARGET_TRAITS_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_AMP)
#include <hpx/compute/amp/config.hpp>
#include <hpx/compute/traits/access_target.hpp>
#include <hpx/compute/amp/target.hpp>


namespace hpx { namespace compute { namespace traits
{
    template <>
    struct access_target<amp::target>
    {
        typedef amp::target target_type;

        template <typename T>
        static T read(amp::target const& tgt, T const* t)
        {
#if defined(__COMPUTE__ACCELERATOR__)
            return *t;
#else
            T tmp;
            //cudaMemcpyAsync(&tmp, t, sizeof(T), cudaMemcpyDeviceToHost,
            //    tgt.native_handle().get_stream());
            //tgt.synchronize();
            //Concurrency::copy()
            std::cout << "READ_TRAITS" << '\n';
            return tmp;
#endif
        }

        template <typename T>
        static void write(amp::target const& tgt, T* dst, T const* src)
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
