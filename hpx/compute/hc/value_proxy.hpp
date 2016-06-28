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
#include <hpx/traits/is_value_proxy.hpp>

#include <type_traits>

namespace hpx { namespace compute { namespace hc
{
    template <typename T>
    class value_proxy
    {
        typedef typename T::value_type value_type;
        typedef traits::access_target<hc::target> access_target;
    public:

        value_proxy(T *p, hc::target & tgt) HPX_NOEXCEPT
          : p_(p)
          , target_(&tgt)
        {}

        value_proxy(value_proxy const& other)
          : p_(other.p_)
          , target_(other.target_)
        {}

        value_proxy& operator=(value_type const& t)
        {
            access_target::write(*target_, p_, &t);
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

        operator value_type() const
        {
            return access_target::read(*target_, p_);
        }

        T* device_ptr() const HPX_NOEXCEPT
        {
            return p_;
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
        T* p_;
        hc::target* target_;
    };


    template <typename T>
    class value_proxy<const T>
    {
        typedef traits::access_target<hc::target> access_target;
        typedef typename T::value_type value_type;
    public:
        typedef T const proxy_type;

        value_proxy(T *p, hc::target & tgt) HPX_NOEXCEPT
            : p_(p)
            , target_(tgt)
        {}

        value_proxy(value_proxy<T> const& other)
            : p_(other.device_ptr())
            , target_(other.target())
        {}

        operator value_type() const
        {
            return access_target::read(target_, p_);
        }

        T* device_ptr() const HPX_NOEXCEPT
        {
            return p_;
        }

        hc::target& target() const HPX_NOEXCEPT
        {
            return *target_;
        }

        const value_type * operator->() const HPX_NOEXCEPT
        {
           return p_;
        }
    private:
        T* p_;
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
