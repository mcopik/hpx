///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_SYCL_VALUE_PROXY_HPP
#define HPX_COMPUTE_SYCL_VALUE_PROXY_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SYCL)

#include <hpx/compute/sycl/config.hpp>
#include <hpx/compute/sycl/traits/access_target.hpp>
#include <hpx/traits/is_value_proxy.hpp>

#include <type_traits>

namespace hpx { namespace compute { namespace sycl
{
    template <typename T>
    class value_proxy
    {
        typedef
            traits::access_target<sycl::target> access_target;

    public:

        value_proxy(buffer_t<T> * buffer, uint64_t pos) HPX_NOEXCEPT
          : buffer_(buffer)
          , pos_(pos)
        {}

        value_proxy(value_proxy const& other)
          : buffer_(other.buffer_)
          , pos_(other.pos_)
        {}

        value_proxy& operator=(T const& t)
        {
            access_target::write(buffer_, pos_, t);
            return *this;
        }

        value_proxy& operator=(value_proxy const& other)
        {
            buffer_ = other.buffer_;
            pos_ = other.pos_;
            return *this;
        }

        operator T() const
        {
            return access_target::read(buffer_, pos_);
        }

        buffer_t<T> * device_data() const
        {
            return buffer_;
        }

    private:
        buffer_t<T> * buffer_;
        uint64_t pos_;
    };
}}}

namespace hpx { namespace traits {
    template <typename T>
    struct is_value_proxy<hpx::compute::sycl::value_proxy<T>>
      : std::true_type
    {};
}}

#endif
#endif
