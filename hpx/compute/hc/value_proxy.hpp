///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_HC_VALUE_PROXY_HPP
#define HPX_COMPUTE_HC_VALUE_PROXY_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_HC)

#include <hpx/compute/hc/config.hpp>
#include <hpx/compute/hc/target.hpp>
#include <hpx/compute/hc/traits/access_target.hpp>
#include <hpx/compute/hc/detail/buffer_proxy.hpp>
#include <hpx/traits/is_value_proxy.hpp>

#include <type_traits>

namespace hpx { namespace compute { namespace hc
{
    template <typename T>
    class value_proxy;

    template <typename T>
    class value_proxy< detail::buffer_proxy<T> >
    {
        typedef detail::buffer_proxy<T> proxy_type;
        typedef traits::access_target<hc::target> access_target;
    public:

        value_proxy(proxy_type *p, hc::target & tgt) HPX_NOEXCEPT
          : p_(p)
          , target_(&tgt)
        {}

        value_proxy(value_proxy const& other)
          : p_(other.p_)
          , target_(other.target_)
        {}

        value_proxy& operator=(T const& t)
        {
            (*p_) = t;
            return *this;
        }

        //value_proxy& operator=(T const& t)
        //{
         //   access_target::write(*target_, p_, &t);
          //  return *this;
        //}

        value_proxy& operator=(value_proxy const& other)
        {
            p_ = other.p_;
            target_ = other.target_;
            return *this;
        }

        operator T() const
        {
            return *p_;
        }

        proxy_type * ptr() const HPX_NOEXCEPT
        {
            return p_;
        }

        T* device_ptr() const HPX_NOEXCEPT
        {
            return p_->device_ptr();
        }

        hc::target& target() const HPX_NOEXCEPT
        {
            return *target_;
        }

/*        value_proxy<value_type> operator*() const HPX_NOEXCEPT
        {
            return value_proxy<value_type>(

        value_type * operator->() const HPX_NOEXCEPT
        {
           return p_;
        }*/
    private:
        proxy_type * p_;
        hc::target* target_;
    };


    template <typename T>
    class value_proxy<const detail::buffer_proxy<T>>
    {
        typedef traits::access_target<hc::target> access_target;
        typedef const detail::buffer_proxy<T> proxy_type;
    public:

        value_proxy(proxy_type *p, hc::target & tgt) HPX_NOEXCEPT
            : p_(p)
            , target_(tgt)
        {}

        /// Required for conversion between value_proxy<T> ->
        /// value_proxy<const T>
        value_proxy(value_proxy<detail::buffer_proxy<T>> const& other)
            : p_(other.ptr())
            , target_(other.target())
        {}

        operator T() const
        {
            return access_target::read(target_, p_);
        }

        T* device_ptr() const HPX_NOEXCEPT
        {
            return p_->device_ptr();
        }

        hc::target& target() const HPX_NOEXCEPT
        {
            return *target_;
        }

        const T * operator->() const HPX_NOEXCEPT
        {
           return p_;
        }
    private:
        proxy_type * p_;
        hc::target& target_;
    };
}}}

namespace hpx { namespace traits {
    template <typename T>
    struct is_value_proxy<hpx::compute::hc::value_proxy<T>>
      : std::true_type
    {};
}}

#endif
#endif
