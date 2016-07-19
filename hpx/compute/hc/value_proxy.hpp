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
    class value_proxy
    {
        typedef buffer_t<T> proxy_type;
        typedef traits::access_target<hc::target> access_target;
    public:

        value_proxy(proxy_type * buffer, T * p) HPX_NOEXCEPT
          : buffer_(buffer),
            pos_(p - buffer_->accelerator_pointer()),
            buffer_view_(*buffer)
        {}

        value_proxy(value_proxy const& other)
          : buffer_(other.buffer_),
            pos_(other.pos_),
            buffer_view_(other.buffer_view_)
        {}

        ~value_proxy()
        {
            std::cout << "TRANSFER" << std::endl;
            // Buffer is not catched in the p_f_e,
            // hence we need to ensure manually that updates
            // in CPU cache are sent to the device.
            //
            // Value proxy should be destroyed immediately after use, because
            // it should exist only as a temporary object. Destructor will
            // perform the copy back.
            //
            // Currently there's no way to copy a single element,
            // a full copy has to be performed.
            buffer_->internal().synchronize(true);
        }

        value_proxy& operator=(T const& t)
        {
            buffer_view_[pos_] = t;
            return *this;
        }

        value_proxy& operator=(value_proxy const& other)
        {
            buffer_ = other.buffer_;
            pos_ = other.pos_;
            buffer_view_ = other.buffer_view_;
            return *this;
        }

        operator T() const
        {
            return buffer_view_[pos_];
        }

        operator T&()
        {
            return buffer_view_[pos_];
        }

        proxy_type * buffer() const HPX_NOEXCEPT
        {
            return buffer_;
        }

        std::ptrdiff_t pos() const HPX_NOEXCEPT
        {
            return pos_;
        }

        buffer_acc_t<T> buffer_at_pos(std::ptrdiff_t size = 0) const HPX_NOEXCEPT
        {
            if (pos_ > 0 && size > 0) {
                return buffer_->section(index<1>(pos_),
                        global_size<1>(size));
            } else if (size > 0) {
                return buffer_->section(global_size<1>(size));
            } else if (pos_ > 0) {
                return buffer_->section(index<1>(pos_));
            } else {
                return buffer_view_;
            }
        }
    private:
        proxy_type * buffer_;
        std::ptrdiff_t pos_;
        buffer_acc_t<T> buffer_view_;
    };


    template <typename T>
    class value_proxy<const T>
    {
        typedef const buffer_t<T> proxy_type;
        typedef traits::access_target<hc::target> access_target;
    public:

        value_proxy(proxy_type * buffer, T * p) HPX_NOEXCEPT
          :
            buffer_(buffer),
            pos_(p - buffer_->accelerator_pointer()),
            buffer_view_(*buffer)
        {}

        value_proxy(const value_proxy<T> & other)
          :
            buffer_(other.buffer()),
            pos_(other.pos()),
            buffer_view_(*buffer_)
        {}

        value_proxy(const value_proxy & other)
          :
            buffer_(other.buffer_),
            pos_(other.pos_),
            buffer_view_(other.buffer_view_)
        {}

        value_proxy& operator=(T const& t)
        {
            //TODO
            buffer_view_[pos_] = t;
            return *this;
        }

        //value_proxy& operator=(T const& t)
        //{
         //   access_target::write(*target_, p_, &t);
          //  return *this;
        //}

        value_proxy& operator=(value_proxy const& other)
        {
            buffer_ = other.buffer_;
            pos_ = other.pos_;
            buffer_view_ = other.buffer_view_;
            return *this;
        }

        operator T() const
        {
            return buffer_view_[pos_];
        }

        proxy_type * ptr() const HPX_NOEXCEPT
        {
            return buffer_;
        }

        //T* device_ptr() const HPX_NOEXCEPT
        //{
        //    return p_;
        //}

        //hc::target& target() const HPX_NOEXCEPT
        //{
        //    return *target_;
        //}

        buffer_acc_t<T> buffer_at_pos(std::ptrdiff_t size = 0) const HPX_NOEXCEPT
        {
            if (pos_ > 0 && size > 0) {
                return buffer_->section(index<1>(pos_),
                        global_size<1>(size));
            } else if (size > 0) {
                return buffer_->section(global_size<1>(size));
            } else if (pos_ > 0) {
                return buffer_->section(index<1>(pos_));
            } else {
                return buffer_view_;
            }
        }
/*        value_proxy<value_type> operator*() const HPX_NOEXCEPT
        {
            return value_proxy<value_type>(

        value_type * operator->() const HPX_NOEXCEPT
        {
           return p_;
        }*/
    private:
        proxy_type *buffer_;
        std::ptrdiff_t pos_;
        buffer_acc_t<const T> buffer_view_;
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
