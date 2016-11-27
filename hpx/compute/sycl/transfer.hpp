//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_SYCL_TRANSFER_HPP
#define HPX_COMPUTE_SYCL_TRANSFER_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SYCL)

#include <hpx/parallel/util/transfer.hpp>
#include <hpx/traits/pointer_category.hpp>

#include <hpx/compute/cuda/allocator.hpp>
#include <hpx/compute/detail/iterator.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

#include <SYCL/accessor.h>

namespace hpx { namespace traits
{
    // Allow for matching of iterator<T const> to iterator<T> while calculating
    // pointer category.
    template <typename T>
    struct remove_const_iterator_value_type<
        compute::detail::iterator<T const, compute::sycl::allocator<T> >
    >
    {
        typedef compute::detail::iterator<T, compute::sycl::allocator<T> > type;
    };

    ///////////////////////////////////////////////////////////////////////////

    struct sycl_pointer_tag : general_pointer_tag {};
    struct sycl_pointer_tag_to_host : sycl_pointer_tag {};
    struct sycl_pointer_tag_to_device : sycl_pointer_tag {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct pointer_category<
        compute::detail::iterator<T, compute::sycl::allocator<T> >,
        compute::detail::iterator<T, compute::sycl::allocator<T> >
    >
    {
        typedef sycl_pointer_tag type;
    };

    template <typename Source, typename T>
    struct pointer_category<
        Source,
        compute::detail::iterator<T, compute::sycl::allocator<T> >,
        typename std::enable_if<
           !std::is_same<
                Source,
                compute::detail::iterator<T, compute::sycl::allocator<T> >
            >::value
        >::type
    >
    {
        static_assert(std::is_same<
                typename hpx::util::decay<T>::type,
                typename std::iterator_traits<Source>::value_type
            >::value, "The value types of the iterators must match");

        typedef sycl_pointer_tag_to_device type;
    };

    template <typename T, typename U, typename Dest>
    struct pointer_category<
        compute::detail::iterator<T, compute::sycl::allocator<U> >,
        Dest,
        typename std::enable_if<
           !std::is_same<
                Dest,
                compute::detail::iterator<T, compute::sycl::allocator<U> >
            >::value
        >::type
    >
    {
        static_assert(std::is_same<
                typename hpx::util::decay<T>::type,
                typename std::iterator_traits<Dest>::value_type
            >::value, "The value types of the iterators must match");

        typedef sycl_pointer_tag_to_host type;
    };
}}

namespace hpx { namespace parallel { namespace util { namespace detail
{
    //FIXME: do we need an implementation used inside GPU kernel?
    template <typename Dummy>
    struct copy_helper<hpx::traits::sycl_pointer_tag, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static std::pair<InIter, OutIter>
        call(InIter first, InIter last, OutIter dest)
        {
            return std::make_pair(last, dest);
        }
    };

    template <typename Dummy>
    struct copy_helper<
        hpx::traits::sycl_pointer_tag_to_host, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static std::pair<InIter, OutIter>
        call(InIter first, InIter last, OutIter dest)
        {
            auto host_access = (*first).device_data()->template
                get_access<
                    cl::sycl::access::mode::read,
                    cl::sycl::access::target::host_buffer
                >();
            std::size_t count = std::distance(first, last);
            auto ptr = host_access.get_pointer() + (*first).pos();
            dest = std::copy(ptr, ptr + count, dest);
            return std::make_pair(last, dest);
        }
    };

    template <typename Dummy>
    struct copy_helper<
        hpx::traits::sycl_pointer_tag_to_device, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static std::pair<InIter, OutIter>
        call(InIter first, InIter last, OutIter dest)
        {
            auto host_access = (*dest).device_data()->template
                get_access<
                    cl::sycl::access::mode::write,
                    cl::sycl::access::target::host_buffer
                >();
            std::copy(first, last, host_access.get_pointer() + (*dest).pos());
            std::size_t count = std::distance(first, last);
            std::advance(dest, count);
            return std::make_pair(last, dest);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Dummy>
    struct copy_n_helper<
        hpx::traits::sycl_pointer_tag, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static std::pair<InIter, OutIter>
        call(InIter first, std::size_t count, OutIter dest)
        {
            return std::make_pair(first, dest);
        }
    };

    template <typename Dummy>
    struct copy_n_helper<
        hpx::traits::sycl_pointer_tag_to_host, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static std::pair<InIter, OutIter>
        call(InIter first, std::size_t count, OutIter dest) {
            try {
            auto host_access = (*first).device_data()->template
                    get_access<
                    cl::sycl::access::mode::read,
                    cl::sycl::access::target::host_buffer
            >();
            auto ptr = host_access.get_pointer() + (*first).pos();
            dest = std::copy(ptr, ptr + count, dest);
            std::advance(first, count);
            } catch (...) {
                std::cout << "exception" << std::endl;
            }
            return std::make_pair(first, dest);
        }
    };

    template <typename Dummy>
    struct copy_n_helper<
        hpx::traits::sycl_pointer_tag_to_device, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static std::pair<InIter, OutIter>
        call(InIter first, std::size_t count, OutIter dest)
        {
            auto host_access = (*dest).device_data()->template
                    get_access<
                    cl::sycl::access::mode::write,
                    cl::sycl::access::target::host_buffer
            >();
            std::copy(first, first + count, host_access.get_pointer() + (*dest).pos());
            std::advance(first, count);
            std::advance(dest, count);
            return std::make_pair(first, dest);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for copy-synchronize operations
    template <typename Dummy>
    struct copy_synchronize_helper<
        hpx::traits::sycl_pointer_tag, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static void
        call(InIter const&, OutIter const& dest)
        {
            dest.target().synchronize();
        }
    };

    template <typename Dummy>
    struct copy_synchronize_helper<
        hpx::traits::sycl_pointer_tag_to_host, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static void
        call(InIter const& first, OutIter const&)
        {
            first.target().synchronize();
        }
    };

    template <typename Dummy>
    struct copy_synchronize_helper<
        hpx::traits::sycl_pointer_tag_to_device, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static void
        call(InIter const&, OutIter const& dest)
        {
            dest.target().synchronize();
        }
    };

}}}}

#endif
#endif
