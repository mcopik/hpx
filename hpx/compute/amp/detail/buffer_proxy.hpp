//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#ifndef HPX_BUFFER_PROXY_HPP
#define HPX_BUFFER_PROXY_HPP

#if defined(HPX_HAVE_AMP)

#include <hpx/compute/amp/traits/access_target.hpp>

#include <amp.h>

namespace hpx { namespace compute { namespace amp
{
    template <typename T>
    class buffer_proxy
    {
        typedef Concurrency::array<T, 1> buffer;
        typedef std::size_t size_type;
    public:
        /// Initialize pointer with begin position of device data
        buffer_proxy(buffer * device_buffer) HPX_NOEXCEPT
            : device_buffer_(device_buffer), p_(device_buffer->data())
        {}

        buffer_proxy(buffer * device_buffer, T * p) HPX_NOEXCEPT
            : device_buffer_(device_buffer), p_(p)
        {}

        buffer_proxy(buffer_proxy const& other)
            : device_buffer_(other.device_buffer_), p_(other.p_)
        {}

        ~buffer_proxy()
        {
            delete device_buffer_;
        }

        operator T() const
        {
            return access_target::read(*target_, p_);
        }

        T& operator*() const
        {
            return *p_;
        }

        buffer_proxy<T> operator+(size_type pos)
        {
            return buffer_proxy(device_buffer_, p_ + pos);
        }

        T& operator++()
        {
            ++p_;
            return *this;
        }

        T& operator--()
        {
            --p_;
            return *this;
        }

    private:
        buffer* device_buffer_;
        T * p_;
    };
}}}

#endif //HPX_BUFFER_PROXY_HPP
