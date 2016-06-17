///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Marcin COpik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_AMP_VALUE_PROXY_HPP
#define HPX_COMPUTE_AMP_VALUE_PROXY_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_AMP) && defined(__CUDACC__)

#include <hpx/compute/amp/target.hpp>
#include <hpx/compute/amp/traits/access_target.hpp>
#include <hpx/traits/is_value_proxy.hpp>

#include <type_traits>

namespace hpx { namespace compute { namespace amp
{
    template <typename T>
    class value_proxy
    {
        typedef traits::access_target<amp::target> access_target;
    public:

        value_proxy(T *p, amp::target & tgt) HPX_NOEXCEPT
          : p_(p)
          , target_(&tgt)
        {}

        value_proxy(value_proxy const& other)
          : p_(other.p_)
          , target_(other.target_)
        {}

        value_proxy& operator=(T const& t)
        {
            access_target::write(*target_, p_, &t);
            return *this;
        }

        value_proxy& operator=(value_proxy const& other)
        {
            p_ = other.p_;
            target_ = other.target_;
            return *this;
        }

        operator T() const
        {
            return access_target::read(*target_, p_);
        }

        T* device_ptr() const HPX_NOEXCEPT
        {
            return p_;
        }

        amp::target& target() const HPX_NOEXCEPT
        {
            return *target_;
        }

    private:
        T* p_;
        amp::target* target_;
    };


    template <typename T const>
    class value_proxy
    {
        typedef traits::access_target<amp::target> access_target;
    public:

        value_proxy(T *p, amp::target & tgt) HPX_NOEXCEPT
          : p_(p)
          , target_(&tgt)
        {}

        value_proxy(value_proxy const& other)
          : p_(other.p_)
          , target_(other.target_)
        {}

        value_proxy& operator=(T const& t)
        {
            access_target::write(*target_, p_, &t);
            return *this;
        }

        value_proxy& operator=(value_proxy const& other)
        {
            p_ = other.p_;
            target_ = other.target_;
            return *this;
        }

        operator T() const
        {
            return access_target::read(*target_, p_);
        }

        T* device_ptr() const HPX_NOEXCEPT
        {
            return p_;
        }

        amp::target& target() const HPX_NOEXCEPT
        {
            return *target_;
        }

    private:
        T* p_;
        amp::target* target_;
    };
}}}

namespace hpx { namespace traits {
    template <typename T>
    struct is_value_proxy<hpx::compute::amp::value_proxy<T>>
      : std::true_type
    {};
}}

#endif
#endif
